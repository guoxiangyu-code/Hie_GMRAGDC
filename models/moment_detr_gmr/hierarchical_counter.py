"""Hierarchical adaptive moment counting for generalized moment retrieval.

HieA2G predicts ``{0, 1, 2, 3, 3+}`` with one classifier.  Soccer-GMR has a
large null class and very few ``4+`` examples, so a flat five-way classifier is
prone to the null collapse observed in the earlier Temporal-AGC pilot.  This
module preserves HieA2G's inference semantics while factorizing the posterior:

    P(N=0) = 1 - P(exists)
    P(N=k) = P(exists) P(N=k | exists),  k in {1,2,3,4+}.

Only positive samples supervise the conditional count head.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(dtype=x.dtype)[..., None]
    return (x * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


class HierarchicalMomentCounter(nn.Module):
    """Predict existence (or its residual) and positive-conditional count."""

    def __init__(
        self,
        hidden_dim: int,
        dropout: float = 0.1,
        detach_query_scores: bool = True,
    ):
        super().__init__()
        self.detach_query_scores = bool(detach_query_scores)
        # text mean, video mean, query weighted mean, query weighted sum,
        # query max, and scalar soft count.
        input_dim = hidden_dim * 5 + 1
        self.project = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.exist_head = nn.Linear(hidden_dim, 1)
        self.count_head = nn.Linear(hidden_dim, 4)  # 1, 2, 3, 4+
        self.ordinal_head = nn.Linear(hidden_dim, 3)  # N>=2, N>=3, N>=4

    def forward(
        self,
        decoder_queries: torch.Tensor,
        pred_logits: torch.Tensor,
        text_memory: torch.Tensor,
        text_mask: torch.Tensor,
        video_memory: torch.Tensor,
        video_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        foreground = pred_logits.softmax(dim=-1)[..., 0]
        weights = foreground.detach() if self.detach_query_scores else foreground
        denominator = weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        query_mean = (decoder_queries * weights[..., None]).sum(dim=1) / denominator
        query_sum = (decoder_queries * weights[..., None]).sum(dim=1)
        query_max = decoder_queries.max(dim=1).values
        soft_count = weights.sum(dim=1, keepdim=True)
        representation = self.project(torch.cat([
            masked_mean(text_memory, text_mask),
            masked_mean(video_memory, video_mask),
            query_mean,
            query_sum,
            query_max,
            soft_count,
        ], dim=-1))
        return {
            "counter_representation": representation,
            "pred_exist_logits": self.exist_head(representation).squeeze(-1),
            "pred_positive_count_logits": self.count_head(representation),
            "pred_count_ordinal_logits": self.ordinal_head(representation),
            "pred_soft_count": foreground.sum(dim=1),
        }


def hierarchical_count_probabilities(outputs: dict[str, torch.Tensor]) -> torch.Tensor:
    """Return a normalized ``[P(0), P(1), P(2), P(3), P(4+)]`` tensor."""
    p_exist = (
        1.0 - torch.sigmoid(outputs["pred_zero_logits"])
        if "pred_zero_logits" in outputs
        else torch.sigmoid(outputs["pred_exist_logits"])
    )
    conditional = outputs["pred_positive_count_logits"].softmax(dim=-1)
    return torch.cat([(1.0 - p_exist)[:, None], p_exist[:, None] * conditional], dim=1)


def _supervised_contrastive_loss(
    representations: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Positive-only, in-batch supervised contrastive loss."""
    if representations.shape[0] < 2:
        return representations.sum() * 0.0
    features = F.normalize(representations, dim=-1)
    logits = features @ features.t() / float(temperature)
    eye = torch.eye(logits.shape[0], dtype=torch.bool, device=logits.device)
    same = labels[:, None].eq(labels[None]) & ~eye
    valid_anchor = same.any(dim=1)
    if not valid_anchor.any():
        return representations.sum() * 0.0

    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    exp_logits = logits.exp() * (~eye).to(logits.dtype)
    log_prob = logits - exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12).log()
    mean_positive = (log_prob * same.to(log_prob.dtype)).sum(dim=1) \
        / same.sum(dim=1).clamp_min(1).to(log_prob.dtype)
    return -mean_positive[valid_anchor].mean()


def hierarchical_counter_losses(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    positive_count_weights: torch.Tensor | None = None,
    contrastive_temperature: float = 0.1,
) -> dict[str, torch.Tensor]:
    """Compute factorized existence, conditional-count and consistency losses."""
    exist = targets["exist_label"].float().view(-1)
    count = targets["count_label"].long().view(-1).clamp(max=4)
    raw_count = targets.get("raw_count_label", count).long().view(-1)
    exist_logits = outputs["pred_exist_logits"].view(-1)
    positive = count > 0
    raw_exist_loss = F.binary_cross_entropy_with_logits(
        exist_logits, exist, reduction="none"
    )
    if positive_count_weights is not None and positive.any():
        class_weights = positive_count_weights.to(
            device=exist_logits.device, dtype=exist_logits.dtype
        )
        # Preserve unit weight for null and single-moment queries while giving
        # rarer multi-moment positives the same relative long-tail emphasis as
        # the conditional count objective. Normalizing by the realized sample
        # weights keeps the overall existence-loss scale stable.
        relative_weights = class_weights / class_weights[0].clamp_min(1e-8)
        sample_weights = torch.ones_like(raw_exist_loss)
        sample_weights[positive] = relative_weights[count[positive] - 1]
        loss_exist = (raw_exist_loss * sample_weights).sum() \
            / sample_weights.sum().clamp_min(1.0)
    else:
        loss_exist = raw_exist_loss.mean()

    zero = exist_logits.sum() * 0.0
    if positive.any():
        positive_targets = count[positive] - 1
        weights = positive_count_weights
        if weights is not None:
            weights = weights.to(
                device=exist_logits.device,
                dtype=outputs["pred_positive_count_logits"].dtype,
            )
        loss_count = F.cross_entropy(
            outputs["pred_positive_count_logits"][positive],
            positive_targets,
            weight=weights,
        )

        thresholds = torch.arange(2, 5, device=count.device)[None]
        ordinal_targets = (count[positive, None] >= thresholds).to(exist_logits.dtype)
        loss_ordinal = F.binary_cross_entropy_with_logits(
            outputs["pred_count_ordinal_logits"][positive], ordinal_targets
        )
        loss_contrastive = _supervised_contrastive_loss(
            outputs["counter_representation"][positive],
            positive_targets,
            temperature=contrastive_temperature,
        )

        closed_count = positive & (raw_count <= 4)
        if closed_count.any():
            conditional = outputs["pred_positive_count_logits"][closed_count].softmax(dim=-1)
            class_values = torch.arange(1, 5, device=count.device, dtype=conditional.dtype)
            expected_count = (conditional * class_values[None]).sum(dim=-1)
            soft_count = outputs["pred_soft_count"][closed_count]
            # The already-trained detector supplies detached count evidence to
            # the new cardinality head; a random count head must not pull the
            # localization logits away from their warm-start solution.
            loss_consistency = F.smooth_l1_loss(expected_count, soft_count.detach())
        else:
            loss_consistency = zero
    else:
        loss_count = zero
        loss_ordinal = zero
        loss_contrastive = zero
        loss_consistency = zero

    return {
        "loss_exist": loss_exist,
        "loss_count": loss_count,
        "loss_count_ordinal": loss_ordinal,
        "loss_count_contrastive": loss_contrastive,
        "loss_count_consistency": loss_consistency,
    }


def inverse_sqrt_positive_count_weights(class_counts: list[int]) -> torch.Tensor:
    """Stable long-tail weights for positive classes ``1/2/3/4+``."""
    counts = torch.as_tensor(class_counts, dtype=torch.float32).clamp_min(1.0)
    weights = counts.rsqrt()
    return weights / weights.mean()
