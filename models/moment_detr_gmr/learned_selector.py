"""Learned null verification and same-event-aware set selection.

The legacy DETR set decoder remains untouched.  This module contains the new
components used by the staged ablations: an independent ``P(N=0)`` verifier,
a symmetric pairwise same-event head, their losses, and count-soft MMR
selection/fusion helpers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from models.moment_detr_gmr.utils.span_utils import span_cxw_to_xx


def _pairwise_geometry(spans_xx: torch.Tensor, eps: float = 1e-6) -> dict[str, torch.Tensor]:
    starts, ends = spans_xx[..., 0], spans_xx[..., 1]
    lengths = (ends - starts).clamp_min(eps)
    centers = 0.5 * (starts + ends)
    intersection = (
        torch.minimum(ends[:, :, None], ends[:, None, :])
        - torch.maximum(starts[:, :, None], starts[:, None, :])
    ).clamp_min(0)
    union = lengths[:, :, None] + lengths[:, None, :] - intersection
    enclosing = (
        torch.maximum(ends[:, :, None], ends[:, None, :])
        - torch.minimum(starts[:, :, None], starts[:, None, :])
    ).clamp_min(eps)
    minimum_length = torch.minimum(lengths[:, :, None], lengths[:, None, :])
    maximum_length = torch.maximum(lengths[:, :, None], lengths[:, None, :])
    return {
        "iou": intersection / union.clamp_min(eps),
        "iom": intersection / minimum_length.clamp_min(eps),
        "center": (centers[:, :, None] - centers[:, None, :]).abs(),
        "start": (starts[:, :, None] - starts[:, None, :]).abs(),
        "end": (ends[:, :, None] - ends[:, None, :]).abs(),
        "duration_ratio": minimum_length / maximum_length.clamp_min(eps),
        "diou_distance": (
            (centers[:, :, None] - centers[:, None, :]) / enclosing
        ).square(),
    }


def _soft_span_pool(
    video: torch.Tensor,
    video_mask: torch.Tensor,
    spans_xx: torch.Tensor,
    temperature: float = 0.03,
) -> torch.Tensor:
    """Differentiably pool video frames inside every normalized span."""
    batch, length, _ = video.shape
    positions = (
        torch.arange(length, device=video.device, dtype=video.dtype) + 0.5
    ) / max(length, 1)
    positions = positions.view(1, 1, length)
    starts = spans_xx[..., 0:1]
    ends = spans_xx[..., 1:2]
    weights = torch.sigmoid((positions - starts) / temperature)
    weights = weights * torch.sigmoid((ends - positions) / temperature)
    weights = weights * video_mask[:, None, :].to(weights.dtype)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    return torch.einsum("bql,bld->bqd", weights, video)


class IndependentZeroVerifier(nn.Module):
    """Predict null from rich evidence independently of the existence logit."""

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        # representation plus score/quality/geometry distribution statistics
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim + 11),
            nn.Linear(hidden_dim + 11, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        representation: torch.Tensor,
        pred_logits: torch.Tensor,
        pred_spans: torch.Tensor,
        quality_logits: torch.Tensor | None,
    ) -> torch.Tensor:
        foreground = pred_logits.softmax(dim=-1)[..., 0]
        quality = (
            torch.sigmoid(quality_logits)
            if quality_logits is not None else torch.ones_like(foreground)
        )
        spans_xx = span_cxw_to_xx(pred_spans).clamp(0, 1)
        geometry = _pairwise_geometry(spans_xx)
        query_count = max(foreground.shape[1], 1)
        normalized = foreground / foreground.sum(dim=1, keepdim=True).clamp_min(1e-6)
        entropy = -(normalized * normalized.clamp_min(1e-8).log()).sum(dim=1)
        entropy = entropy / math.log(query_count) if query_count > 1 else entropy * 0
        eye = torch.eye(
            query_count, dtype=torch.bool, device=foreground.device
        )[None]
        off_diagonal_iou = geometry["iou"].masked_fill(eye, 0.0)
        centers = spans_xx.mean(dim=-1)
        sorted_centers = centers.sort(dim=1).values
        gaps = sorted_centers[:, 1:] - sorted_centers[:, :-1]
        stats = torch.stack([
            foreground.max(dim=1).values,
            foreground.mean(dim=1),
            foreground.sum(dim=1),
            foreground.std(dim=1, unbiased=False),
            entropy,
            quality.max(dim=1).values,
            quality.mean(dim=1),
            off_diagonal_iou.max(dim=2).values.max(dim=1).values,
            off_diagonal_iou.sum(dim=(1, 2)) / max(query_count * (query_count - 1), 1),
            gaps.min(dim=1).values if gaps.shape[1] else foreground.new_zeros(foreground.shape[0]),
            gaps.mean(dim=1) if gaps.shape[1] else foreground.new_zeros(foreground.shape[0]),
        ], dim=-1)
        return self.net(torch.cat([representation, stats], dim=-1)).squeeze(-1)


class PairwiseSameEventHead(nn.Module):
    """Symmetric head for ``P(query_i, query_j refer to the same event)``."""

    def __init__(
        self,
        hidden_dim: int,
        dropout: float = 0.1,
        detach_inputs: bool = True,
    ):
        super().__init__()
        self.detach_inputs = bool(detach_inputs)
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.video_norm = nn.LayerNorm(hidden_dim)
        # |qi-qj|, qi*qj, |vi-vj|, vi*vj, seven geometry, two query-score
        # terms, and mean/peak frame evidence between the two centers.
        input_dim = hidden_dim * 4 + 11
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        decoder_queries: torch.Tensor,
        pred_logits: torch.Tensor,
        pred_spans: torch.Tensor,
        video_memory: torch.Tensor,
        video_mask: torch.Tensor,
        frame_event_scores: torch.Tensor | None = None,
    ) -> torch.Tensor:
        queries = decoder_queries
        spans = pred_spans
        video = video_memory
        logits = pred_logits
        if self.detach_inputs:
            queries, spans, video, logits = (
                queries.detach(), spans.detach(), video.detach(), logits.detach()
            )
        queries = self.query_norm(queries)
        spans_xx = span_cxw_to_xx(spans).clamp(0, 1)
        local_video = self.video_norm(_soft_span_pool(video, video_mask, spans_xx))
        foreground = logits.softmax(dim=-1)[..., 0]
        geometry = _pairwise_geometry(spans_xx)
        centers = spans_xx.mean(dim=-1)
        interval_left = torch.minimum(centers[:, :, None], centers[:, None, :])
        interval_right = torch.maximum(centers[:, :, None], centers[:, None, :])
        positions = (
            torch.arange(video.shape[1], device=video.device, dtype=video.dtype) + 0.5
        ) / max(video.shape[1], 1)
        between_weights = torch.sigmoid(
            (positions.view(1, 1, 1, -1) - interval_left[..., None]) / 0.03
        ) * torch.sigmoid(
            (interval_right[..., None] - positions.view(1, 1, 1, -1)) / 0.03
        )
        between_weights = between_weights * video_mask[:, None, None, :].to(
            between_weights.dtype
        )
        if frame_event_scores is None:
            frame_evidence = video.new_zeros(video.shape[:2])
        else:
            frame_evidence = torch.sigmoid(
                frame_event_scores.detach() if self.detach_inputs else frame_event_scores
            )
        between_mean = (
            between_weights * frame_evidence[:, None, None, :]
        ).sum(dim=-1) / between_weights.sum(dim=-1).clamp_min(1e-6)
        between_peak = (
            between_weights * frame_evidence[:, None, None, :]
        ).max(dim=-1).values

        query_abs = (queries[:, :, None] - queries[:, None, :]).abs()
        query_product = queries[:, :, None] * queries[:, None, :]
        video_abs = (local_video[:, :, None] - local_video[:, None, :]).abs()
        video_product = local_video[:, :, None] * local_video[:, None, :]
        scalar_features = torch.stack([
            geometry["iou"], geometry["iom"], geometry["center"],
            geometry["start"], geometry["end"], geometry["duration_ratio"],
            geometry["diou_distance"],
            0.5 * (foreground[:, :, None] + foreground[:, None, :]),
            (foreground[:, :, None] - foreground[:, None, :]).abs(),
            between_mean, between_peak,
        ], dim=-1)
        features = torch.cat([
            query_abs, query_product, video_abs, video_product, scalar_features
        ], dim=-1)
        pair_logits = self.net(features).squeeze(-1)
        return 0.5 * (pair_logits + pair_logits.transpose(1, 2))


def independent_zero_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    positive_query_weight: float = 1.0,
) -> torch.Tensor:
    logits = outputs["pred_zero_logits"].reshape(-1)
    exists = targets["exist_label"].to(logits).reshape(-1)
    target_zero = 1.0 - exists
    raw = F.binary_cross_entropy_with_logits(logits, target_zero, reduction="none")
    weights = torch.where(
        exists > 0.5,
        torch.full_like(raw, float(positive_query_weight)),
        torch.ones_like(raw),
    )
    return (raw * weights).sum() / weights.sum().clamp_min(1.0)


def _candidate_assignments(
    predicted_xx: torch.Tensor,
    target_xx: torch.Tensor,
    minimum_iou: float,
    ambiguity_margin: float,
) -> torch.Tensor:
    """Assign every candidate to a GT instance; ``-1`` means ignore."""
    if target_xx.numel() == 0:
        return torch.full(
            (predicted_xx.shape[0],), -1, dtype=torch.long, device=predicted_xx.device
        )
    pred = predicted_xx[:, None, :]
    target = target_xx[None, :, :]
    intersection = (
        torch.minimum(pred[..., 1], target[..., 1])
        - torch.maximum(pred[..., 0], target[..., 0])
    ).clamp_min(0)
    union = (
        (pred[..., 1] - pred[..., 0]).clamp_min(0)
        + (target[..., 1] - target[..., 0]).clamp_min(0)
        - intersection
    ).clamp_min(1e-6)
    overlaps = intersection / union
    best_iou, best_index = overlaps.max(dim=1)
    valid = best_iou >= float(minimum_iou)
    if overlaps.shape[1] > 1:
        top_two = overlaps.topk(k=2, dim=1).values
        valid &= (top_two[:, 0] - top_two[:, 1]) >= float(ambiguity_margin)
    return torch.where(valid, best_index, torch.full_like(best_index, -1))


def pairwise_same_event_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict,
    *,
    assignment_iou: float = 0.3,
    ambiguity_margin: float = 0.05,
    positive_weight: float = 1.0,
    hard_negative_weight: float = 2.0,
) -> torch.Tensor:
    logits = outputs["pred_same_event_logits"]
    predicted = span_cxw_to_xx(outputs["pred_spans"]).clamp(0, 1).detach()
    total = logits.sum() * 0.0
    total_weight = logits.new_zeros(())
    query_count = logits.shape[1]
    upper = torch.triu(
        torch.ones(query_count, query_count, dtype=torch.bool, device=logits.device),
        diagonal=1,
    )
    predicted_geometry = _pairwise_geometry(predicted)["iou"].detach()
    for batch_index, span_target in enumerate(targets["span_labels"]):
        target_spans = span_target["spans"].to(predicted)
        target_xx = span_cxw_to_xx(target_spans).clamp(0, 1)
        assignment = _candidate_assignments(
            predicted[batch_index], target_xx, assignment_iou, ambiguity_margin
        )
        valid = (assignment[:, None] >= 0) & (assignment[None, :] >= 0) & upper
        if not valid.any():
            continue
        labels = assignment[:, None].eq(assignment[None, :]).to(logits.dtype)
        weights = torch.ones_like(labels)
        weights = torch.where(
            labels > 0.5,
            weights * float(positive_weight),
            weights * (1.0 + float(hard_negative_weight) * predicted_geometry[batch_index]),
        )
        raw = F.binary_cross_entropy_with_logits(
            logits[batch_index], labels, reduction="none"
        )
        total = total + (raw * weights * valid).sum()
        total_weight = total_weight + (weights * valid).sum()
    return total / total_weight.clamp_min(1.0)


def combine_two_stage_existence(
    gate_scores: torch.Tensor,
    zero_scores: torch.Tensor,
    localization_scores: torch.Tensor,
    *,
    mode: str = "cascade",
    veto_threshold: float = 0.7,
    localization_threshold: float = 0.2,
) -> torch.Tensor:
    """Continuous score implementing rescue first and cautious veto second."""
    if mode not in {"gate", "zero", "rescue", "cascade"}:
        raise ValueError(f"Unknown two-stage existence mode: {mode}")
    positive_from_zero = 1.0 - zero_scores
    if mode == "gate":
        return gate_scores
    if mode == "zero":
        return positive_from_zero
    rescued = torch.maximum(gate_scores, positive_from_zero)
    if mode == "rescue":
        return rescued
    veto = (
        (zero_scores >= float(veto_threshold))
        & (localization_scores < float(localization_threshold))
    )
    return torch.where(veto, torch.minimum(gate_scores, positive_from_zero), rescued)


def two_stage_accept(
    gate_scores: torch.Tensor,
    zero_scores: torch.Tensor,
    localization_scores: torch.Tensor,
    *,
    gate_threshold: float = 0.3,
    zero_threshold: float = 0.6,
    veto_threshold: float = 0.7,
    localization_threshold: float = 0.2,
) -> torch.Tensor:
    """Exact two-stage decision: agree-null rejects, conflicts pass, cautious veto."""
    if float(veto_threshold) < float(zero_threshold):
        raise ValueError("veto_threshold must be at least zero_threshold")
    stage_one_null = gate_scores < float(gate_threshold)
    stage_two_null = zero_scores >= float(zero_threshold)
    agree_null = stage_one_null & stage_two_null
    cautious_veto = (
        ~stage_one_null
        & (zero_scores >= float(veto_threshold))
        & (localization_scores < float(localization_threshold))
    )
    return ~(agree_null | cautious_veto)


@dataclass(frozen=True)
class SelectionResult:
    selected: list[int]
    marginal_utilities: list[float]


def learned_mmr_select(
    scores: torch.Tensor,
    duplicate_probabilities: torch.Tensor,
    *,
    max_output: int,
    redundancy_lambda: float,
    count_probabilities: torch.Tensor | None = None,
    count_prior_weight: float = 0.0,
    stop_threshold: float = float("-inf"),
) -> SelectionResult:
    """Select candidates with learned redundancy and an entropy-scaled count prior."""
    if scores.ndim != 1 or duplicate_probabilities.shape != (scores.numel(), scores.numel()):
        raise ValueError("scores must be [Q] and duplicate_probabilities [Q,Q]")
    if max_output < 0:
        raise ValueError("max_output must be non-negative")
    if scores.numel() == 0 or max_output == 0:
        return SelectionResult([], [])
    eps = torch.finfo(scores.dtype).eps
    base = torch.logit(scores.clamp(eps, 1.0 - eps))
    beta = float(count_prior_weight)
    conditional = None
    if count_probabilities is not None:
        if count_probabilities.shape != (5,):
            raise ValueError("count_probabilities must be [P0,P1,P2,P3,P4+]")
        conditional = count_probabilities[1:]
        conditional = conditional / conditional.sum().clamp_min(eps)
        entropy = -(conditional * conditional.clamp_min(eps).log()).sum()
        beta *= max(0.0, 1.0 - float(entropy) / math.log(4.0))

    remaining = set(range(scores.numel()))
    selected: list[int] = []
    utilities: list[float] = []
    while remaining and len(selected) < max_output:
        best_index, best_utility = None, None
        for index in sorted(remaining):
            redundancy = (
                max(float(duplicate_probabilities[index, prior]) for prior in selected)
                if selected else 0.0
            )
            utility = float(base[index]) - float(redundancy_lambda) * redundancy
            if conditional is not None and selected:
                old_bucket = min(len(selected), 4) - 1
                new_bucket = min(len(selected) + 1, 4) - 1
                utility += beta * float(
                    conditional[new_bucket].clamp_min(eps).log()
                    - conditional[old_bucket].clamp_min(eps).log()
                )
            if best_utility is None or utility > best_utility:
                best_index, best_utility = index, utility
        assert best_index is not None and best_utility is not None
        # A confirmed non-null query always emits at least one representative.
        if selected and best_utility <= float(stop_threshold):
            break
        selected.append(best_index)
        utilities.append(best_utility)
        remaining.remove(best_index)
    return SelectionResult(selected, utilities)


def cautious_complete_link_fusion(
    spans_xx: torch.Tensor,
    scores: torch.Tensor,
    duplicate_probabilities: torch.Tensor,
    selected: list[int],
    *,
    same_event_threshold: float = 0.8,
    boundary_std_threshold: float = 0.03,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Attach duplicates with complete-link agreement; fuse only stable clusters."""
    if not selected:
        return spans_xx.new_empty((0, 2)), scores.new_empty((0,))
    assigned = set(selected)
    clusters = [[index] for index in selected]
    candidates = [i for i in torch.argsort(scores, descending=True).tolist() if i not in assigned]
    for candidate in candidates:
        compatible = []
        for cluster_index, cluster in enumerate(clusters):
            probabilities = duplicate_probabilities[candidate, cluster]
            if bool((probabilities >= float(same_event_threshold)).all()):
                compatible.append((float(probabilities.mean()), cluster_index))
        if compatible:
            _, cluster_index = max(compatible)
            clusters[cluster_index].append(candidate)
            assigned.add(candidate)

    output_spans, output_scores = [], []
    for medoid, cluster in zip(selected, clusters):
        members = torch.as_tensor(cluster, dtype=torch.long, device=spans_xx.device)
        member_spans = spans_xx[members]
        if len(cluster) > 1 and float(member_spans.std(dim=0, unbiased=False).max()) <= float(boundary_std_threshold):
            weights = scores[members].clamp_min(1e-8)
            fused = (member_spans * weights[:, None]).sum(dim=0) / weights.sum()
            output_spans.append(fused)
        else:
            output_spans.append(spans_xx[medoid])
        output_scores.append(scores[medoid])
    return torch.stack(output_spans), torch.stack(output_scores)


__all__ = [
    "IndependentZeroVerifier", "PairwiseSameEventHead", "SelectionResult",
    "independent_zero_loss", "pairwise_same_event_loss",
    "combine_two_stage_existence", "two_stage_accept", "learned_mmr_select",
    "cautious_complete_link_fusion",
]
