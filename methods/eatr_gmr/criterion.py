"""MR-only EaTR criterion with first-class mixed and all-null support."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .adapter import existence_loss
from .dual_grounding import dual_grounding_losses
from .hierarchical_counter import hierarchical_counter_losses
from .spans import generalized_temporal_iou, generalized_temporal_iou_, span_cxw_to_xx


class SetCriterion(nn.Module):
    """EaTR localization/event losses plus optional GMR existence BCE."""

    def __init__(self, matcher: nn.Module, event_matcher: nn.Module,
                 weight_dict: dict[str, float], eos_coef: float = 0.1,
                 aux_loss: bool = True, use_quality_head: bool = False,
                 use_dual_grounding: bool = False,
                 use_hierarchical_counter: bool = False,
                 mask_null_vmr_loss: bool = False,
                 positive_count_weights: torch.Tensor | None = None,
                 dual_dqa_scale: float = 0.3,
                 dual_eos_temperature: float = 0.07,
                 counter_contrastive_temperature: float = 0.1) -> None:
        super().__init__()
        self.matcher = matcher
        self.event_matcher = event_matcher
        self.weight_dict = dict(weight_dict)
        self.aux_loss = bool(aux_loss)
        self.use_quality_head = bool(use_quality_head)
        self.use_dual_grounding = bool(use_dual_grounding)
        self.use_hierarchical_counter = bool(use_hierarchical_counter)
        self.mask_null_vmr_loss = bool(mask_null_vmr_loss)
        self.dual_dqa_scale = float(dual_dqa_scale)
        self.dual_eos_temperature = float(dual_eos_temperature)
        self.counter_contrastive_temperature = float(counter_contrastive_temperature)
        class_weight = torch.ones(2)
        class_weight[1] = float(eos_coef)
        self.register_buffer("class_weight", class_weight)
        if positive_count_weights is None:
            positive_count_weights = torch.ones(4, dtype=torch.float32)
        self.register_buffer(
            "positive_count_weights",
            torch.as_tensor(positive_count_weights, dtype=torch.float32),
        )

    @staticmethod
    def _src_indices(indices):
        batch = torch.cat([
            torch.full_like(src, sample_index)
            for sample_index, (src, _) in enumerate(indices)
        ])
        source = torch.cat([src for src, _ in indices])
        return batch, source

    @staticmethod
    def _matched_count(indices) -> int:
        return sum(int(source.numel()) for source, _ in indices)

    def _vmr_positive_mask(self, targets: dict, batch_size: int,
                           device: torch.device) -> torch.Tensor:
        if not self.mask_null_vmr_loss:
            return torch.ones(batch_size, dtype=torch.bool, device=device)
        if "exist_label" not in targets:
            raise ValueError(
                "mask_null_vmr_loss requires targets['exist_label']"
            )
        positive = targets["exist_label"].to(device=device).reshape(-1) > 0.5
        if positive.numel() != batch_size:
            raise ValueError(
                "exist_label batch size does not match model output: "
                f"{positive.numel()} != {batch_size}"
            )
        return positive

    @staticmethod
    def _keep_positive_indices(indices, positive: torch.Tensor):
        return [
            (source, target) if bool(positive[index].item())
            else (source[:0], target[:0])
            for index, (source, target) in enumerate(indices)
        ]

    def loss_labels(self, outputs: dict, targets: dict, indices,
                    suffix: str = "") -> dict:
        logits = outputs["pred_logits"]
        target_classes = torch.full(
            logits.shape[:2], 1, dtype=torch.long, device=logits.device
        )
        if self._matched_count(indices):
            target_classes[self._src_indices(indices)] = 0
        # Preserve EaTR/DETR's explicit per-query averaging so ``eos_coef``
        # still downweights an all-background (all-null) batch.
        per_query = F.cross_entropy(
            logits.transpose(1, 2), target_classes,
            weight=self.class_weight, reduction="none",
        )
        if self.mask_null_vmr_loss:
            positive = self._vmr_positive_mask(
                targets, logits.shape[0], logits.device
            )[:, None].expand_as(per_query)
            loss = (per_query * positive).sum() / positive.sum().clamp_min(1)
        else:
            loss = per_query.mean()
        return {f"loss_label{suffix}": loss}

    def loss_spans(self, outputs: dict, targets: dict, indices,
                   suffix: str = "") -> dict:
        pred = outputs["pred_spans"]
        if self._matched_count(indices) == 0:
            zero = pred.sum() * 0.0
            return {f"loss_span{suffix}": zero, f"loss_giou{suffix}": zero}

        src = pred[self._src_indices(indices)]
        target_list = targets["span_labels"]
        dst = torch.cat([
            target["spans"][target_indices]
            for target, (_, target_indices) in zip(target_list, indices)
        ])
        loss_span = F.l1_loss(src, dst)
        giou = generalized_temporal_iou(span_cxw_to_xx(src), span_cxw_to_xx(dst))
        loss_giou = (1.0 - torch.diag(giou)).mean()
        return {f"loss_span{suffix}": loss_span, f"loss_giou{suffix}": loss_giou}

    def loss_events(self, outputs: dict, indices) -> dict:
        pred = outputs["pred_event_spans"]
        if self._matched_count(indices) == 0:
            zero = pred.sum() * 0.0
            return {"loss_event_span": zero, "loss_event_giou": zero}
        src = pred[self._src_indices(indices)]
        dst = torch.cat([
            target[target_indices]
            for target, (_, target_indices) in zip(outputs["pseudo_event_spans"], indices)
        ]).to(src)
        loss_span = F.l1_loss(src, dst)
        giou = generalized_temporal_iou_(span_cxw_to_xx(src), span_cxw_to_xx(dst))
        loss_giou = (1.0 - torch.diag(giou)).mean()
        return {"loss_event_span": loss_span, "loss_event_giou": loss_giou}

    def loss_quality(self, outputs: dict, targets: dict, indices) -> dict:
        """Regress matched temporal IoU and zero quality for unmatched slots."""
        logits = outputs["pred_quality_logits"]
        quality_targets = torch.zeros_like(logits)
        weights = torch.ones_like(logits) * self.class_weight[1]
        if self._matched_count(indices):
            source_index = self._src_indices(indices)
            source_spans = outputs["pred_spans"][source_index]
            target_spans = torch.cat([
                target["spans"][target_indices]
                for target, (_, target_indices) in zip(targets["span_labels"], indices)
            ])
            source_xx = span_cxw_to_xx(source_spans).clamp(0.0, 1.0)
            target_xx = span_cxw_to_xx(target_spans).clamp(0.0, 1.0)
            intersection = (
                torch.minimum(source_xx[:, 1], target_xx[:, 1])
                - torch.maximum(source_xx[:, 0], target_xx[:, 0])
            ).clamp_min(0.0)
            union = (
                (source_xx[:, 1] - source_xx[:, 0]).clamp_min(0.0)
                + (target_xx[:, 1] - target_xx[:, 0]).clamp_min(0.0)
                - intersection
            ).clamp_min(1e-6)
            quality_targets[source_index] = (intersection / union).detach()
            weights[source_index] = 1.0
        raw = F.binary_cross_entropy_with_logits(
            logits, quality_targets, reduction="none"
        )
        if self.mask_null_vmr_loss:
            positive = self._vmr_positive_mask(
                targets, logits.shape[0], logits.device
            )[:, None]
            weights = weights * positive
        return {"loss_quality": (raw * weights).sum() / weights.sum().clamp_min(1.0)}

    def forward(self, outputs: dict, targets: dict) -> dict[str, torch.Tensor]:
        main_outputs = {
            "pred_logits": outputs["pred_logits"],
            "pred_spans": outputs["pred_spans"],
        }
        indices = self.matcher(main_outputs, targets)
        if self.mask_null_vmr_loss:
            positive = self._vmr_positive_mask(
                targets, outputs["pred_logits"].shape[0],
                outputs["pred_logits"].device,
            )
            indices = self._keep_positive_indices(indices, positive)
        losses = {}
        losses.update(self.loss_labels(main_outputs, targets, indices))
        losses.update(self.loss_spans(main_outputs, targets, indices))

        event_indices = self.event_matcher(
            outputs["pred_event_spans"], outputs["pseudo_event_spans"]
        )
        if self.mask_null_vmr_loss:
            event_indices = self._keep_positive_indices(event_indices, positive)
        losses.update(self.loss_events(outputs, event_indices))

        if "pred_exist_logits" in outputs:
            if "exist_label" not in targets:
                raise ValueError("GMR adapter requires exist_label in targets")
            if not self.use_hierarchical_counter:
                losses["loss_exist"] = existence_loss(
                    outputs["pred_exist_logits"], targets["exist_label"]
                )

        if self.use_quality_head:
            losses.update(self.loss_quality(outputs, targets, indices))
        if self.use_dual_grounding:
            sample_mask = None
            if self.mask_null_vmr_loss:
                attention = outputs["dual_phrase_attention"]
                sample_mask = self._vmr_positive_mask(
                    targets, attention.shape[0], attention.device
                )
            losses.update(dual_grounding_losses(
                outputs,
                dqa_scale=self.dual_dqa_scale,
                temperature=self.dual_eos_temperature,
                sample_mask=sample_mask,
            ))
        if self.use_hierarchical_counter:
            losses.update(hierarchical_counter_losses(
                outputs,
                targets,
                positive_count_weights=self.positive_count_weights,
                contrastive_temperature=self.counter_contrastive_temperature,
            ))

        if self.aux_loss:
            for layer_index, aux in enumerate(outputs.get("aux_outputs", [])):
                aux_indices = self.matcher(aux, targets)
                if self.mask_null_vmr_loss:
                    aux_indices = self._keep_positive_indices(aux_indices, positive)
                suffix = f"_{layer_index}"
                losses.update(self.loss_labels(aux, targets, aux_indices, suffix))
                losses.update(self.loss_spans(aux, targets, aux_indices, suffix))
        return losses

    def weighted_loss(self, losses: dict[str, torch.Tensor]) -> torch.Tensor:
        weighted = [
            value * self.weight_dict[name]
            for name, value in losses.items()
            if name in self.weight_dict
        ]
        if not weighted:
            raise ValueError("criterion produced no weighted losses")
        return torch.stack(weighted).sum()
