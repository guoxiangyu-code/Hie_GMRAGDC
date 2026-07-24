"""Quality calibration and independent null verification for Flash-VTG.

The heads in this file deliberately do not consume the legacy existence
logit.  ``FlashIndependentZeroVerifier`` therefore provides a genuinely
independent ``P(N=0)`` signal, while ``FlashCandidateQualityHead`` calibrates
each anchor-free candidate to its temporal IoU with the closest ground-truth
moment.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn


def _as_query_vector(query: torch.Tensor) -> torch.Tensor:
    """Return a ``[B, D]`` query representation for either pooling layout."""
    if query.ndim == 2:
        return query
    if query.ndim != 3:
        raise ValueError("query must have shape [B,D] or [B,L,D]")
    if query.shape[1] == 1:
        return query[:, 0]
    return query.mean(dim=1)


def _masked_video_pool(
    video: torch.Tensor,
    video_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Masked mean/max pooling with finite output for fully padded rows."""
    if video.ndim != 3 or video_mask.ndim != 2:
        raise ValueError("video/video_mask must have shapes [B,L,D]/[B,L]")
    valid = video_mask.to(dtype=video.dtype)
    mean = (video * valid.unsqueeze(-1)).sum(dim=1)
    mean = mean / valid.sum(dim=1, keepdim=True).clamp_min(1.0)
    maximum = video.masked_fill(~video_mask.bool().unsqueeze(-1), float("-inf"))
    maximum = maximum.max(dim=1).values
    maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
    return mean, maximum


def _temporal_iou(
    first: torch.Tensor,
    second: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Pairwise temporal IoU for ``[N,2]`` and ``[M,2]`` xx spans."""
    intersection = (
        torch.minimum(first[:, None, 1], second[None, :, 1])
        - torch.maximum(first[:, None, 0], second[None, :, 0])
    ).clamp_min(0)
    first_length = (first[:, 1] - first[:, 0]).clamp_min(0)
    second_length = (second[:, 1] - second[:, 0]).clamp_min(0)
    union = first_length[:, None] + second_length[None, :] - intersection
    return intersection / union.clamp_min(eps)


def _cxw_to_xx(spans: torch.Tensor) -> torch.Tensor:
    center, width = spans.unbind(dim=-1)
    return torch.stack((center - 0.5 * width, center + 0.5 * width), dim=-1)


class FlashCandidateQualityHead(nn.Module):
    """Predict an IoU-quality logit for every Flash-VTG pyramid candidate."""

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        # A constant initial quality keeps the parent candidate ordering.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        pyramid_features: torch.Tensor | Sequence[torch.Tensor],
    ) -> torch.Tensor:
        if isinstance(pyramid_features, (list, tuple)):
            pyramid_features = torch.cat(list(pyramid_features), dim=1)
        if pyramid_features.ndim != 3:
            raise ValueError("pyramid features must have shape [B,N,D]")
        return self.net(pyramid_features).squeeze(-1)


class FlashIndependentZeroVerifier(nn.Module):
    """Predict ``P(N=0)`` from representations and candidate evidence.

    No existence-head value is accepted by the interface.  The verifier uses
    query/video interactions plus distribution and temporal-geometry
    statistics from Flash-VTG candidates.
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.representation = nn.Sequential(
            nn.LayerNorm(hidden_dim * 5),
            nn.Linear(hidden_dim * 5, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim + 11),
            nn.Linear(hidden_dim + 11, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        # Neutral P(null)=0.5 at warm start.
        nn.init.zeros_(self.classifier[-1].weight)
        nn.init.zeros_(self.classifier[-1].bias)

    def forward(
        self,
        query: torch.Tensor,
        video: torch.Tensor,
        video_mask: torch.Tensor,
        candidate_logits: torch.Tensor,
        candidate_spans_xx: torch.Tensor,
        quality_logits: torch.Tensor | None = None,
        candidate_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        query = _as_query_vector(query).float()
        video_mean, video_max = _masked_video_pool(video.float(), video_mask)
        representation = self.representation(torch.cat([
            query,
            video_mean,
            video_max,
            query * video_mean,
            (query - video_mean).abs(),
        ], dim=-1))

        foreground = candidate_logits
        if foreground.ndim == 3 and foreground.shape[-1] == 1:
            foreground = foreground[..., 0]
        if foreground.ndim != 2:
            raise ValueError("candidate_logits must have shape [B,N] or [B,N,1]")
        foreground = torch.sigmoid(foreground.float())
        quality = (
            torch.sigmoid(quality_logits.float())
            if quality_logits is not None else torch.ones_like(foreground)
        )
        if candidate_spans_xx.shape != (*foreground.shape, 2):
            raise ValueError("candidate_spans_xx must have shape [B,N,2]")

        valid = (
            candidate_mask.bool()
            if candidate_mask is not None else torch.ones_like(foreground, dtype=torch.bool)
        )
        if valid.shape != foreground.shape:
            raise ValueError("candidate_mask must have shape [B,N]")
        count = valid.sum(dim=1).clamp_min(1)
        valid_float = valid.to(foreground)
        masked_foreground = foreground * valid_float
        masked_quality = quality * valid_float

        probability_mass = masked_foreground.sum(dim=1)
        normalized = masked_foreground / probability_mass[:, None].clamp_min(1e-6)
        entropy = -(normalized * normalized.clamp_min(1e-8).log()).sum(dim=1)
        entropy = entropy / count.float().clamp_min(2.0).log()

        spans = candidate_spans_xx.float().clamp(0, 1)
        starts, ends = spans[..., 0], spans[..., 1]
        lengths = (ends - starts).clamp_min(1e-6)
        intersection = (
            torch.minimum(ends[:, :, None], ends[:, None, :])
            - torch.maximum(starts[:, :, None], starts[:, None, :])
        ).clamp_min(0)
        union = lengths[:, :, None] + lengths[:, None, :] - intersection
        pair_iou = intersection / union.clamp_min(1e-6)
        pair_valid = valid[:, :, None] & valid[:, None, :]
        diagonal = torch.eye(
            foreground.shape[1], dtype=torch.bool, device=foreground.device
        )[None]
        off_diagonal = pair_valid & ~diagonal
        pair_iou = pair_iou.masked_fill(~off_diagonal, 0.0)
        pair_count = off_diagonal.sum(dim=(1, 2)).clamp_min(1)

        centers = 0.5 * (starts + ends)
        # Invalid candidates sort last; adjacent differences between valid
        # centers provide a compact event-separation statistic.
        sorted_centers = centers.masked_fill(~valid, float("inf")).sort(dim=1).values
        gaps = sorted_centers[:, 1:] - sorted_centers[:, :-1]
        gap_valid = torch.arange(
            max(foreground.shape[1] - 1, 0), device=foreground.device
        )[None] < (count - 1).clamp_min(0)[:, None]
        if gaps.shape[1]:
            min_gap = gaps.masked_fill(~gap_valid, float("inf")).min(dim=1).values
            min_gap = torch.where(torch.isfinite(min_gap), min_gap, torch.zeros_like(min_gap))
            mean_gap = (gaps.masked_fill(~gap_valid, 0.0).sum(dim=1)
                        / gap_valid.sum(dim=1).clamp_min(1))
        else:
            min_gap = foreground.new_zeros(foreground.shape[0])
            mean_gap = foreground.new_zeros(foreground.shape[0])

        neg_inf = torch.full_like(foreground, float("-inf"))
        foreground_max = torch.where(valid, foreground, neg_inf).max(dim=1).values
        quality_max = torch.where(valid, quality, neg_inf).max(dim=1).values
        foreground_max = torch.where(
            torch.isfinite(foreground_max), foreground_max, torch.zeros_like(foreground_max)
        )
        quality_max = torch.where(
            torch.isfinite(quality_max), quality_max, torch.zeros_like(quality_max)
        )
        foreground_mean = masked_foreground.sum(dim=1) / count
        quality_mean = masked_quality.sum(dim=1) / count
        foreground_variance = (
            ((foreground - foreground_mean[:, None]).square() * valid_float).sum(dim=1)
            / count
        )
        stats = torch.stack([
            foreground_max,
            foreground_mean,
            probability_mass,
            foreground_variance.sqrt(),
            entropy,
            quality_max,
            quality_mean,
            pair_iou.amax(dim=(1, 2)),
            pair_iou.sum(dim=(1, 2)) / pair_count,
            min_gap,
            mean_gap,
        ], dim=-1)
        return self.classifier(torch.cat([representation, stats], dim=-1)).squeeze(-1)


def flash_candidate_quality_loss(
    quality_logits: torch.Tensor,
    candidate_spans_xx: torch.Tensor,
    span_labels: Sequence[dict[str, torch.Tensor]],
    candidate_mask: torch.Tensor | None = None,
    negative_candidate_weight: float = 0.1,
) -> torch.Tensor:
    """Supervise candidate quality with maximum IoU on non-null samples only."""
    if quality_logits.ndim != 2:
        raise ValueError("quality_logits must have shape [B,N]")
    if candidate_spans_xx.shape != (*quality_logits.shape, 2):
        raise ValueError("candidate_spans_xx must have shape [B,N,2]")
    if len(span_labels) != quality_logits.shape[0]:
        raise ValueError("span_labels batch size does not match logits")

    targets = torch.zeros_like(quality_logits)
    weights = torch.zeros_like(quality_logits)
    valid = (
        candidate_mask.bool()
        if candidate_mask is not None else torch.ones_like(quality_logits, dtype=torch.bool)
    )
    for batch_index, label in enumerate(span_labels):
        target_cxw = label["spans"].to(candidate_spans_xx)
        if target_cxw.numel() == 0:
            # Null samples must not enter the localization-quality loss.
            continue
        target_xx = _cxw_to_xx(target_cxw).clamp(0, 1)
        predicted_xx = candidate_spans_xx[batch_index].detach().clamp(0, 1)
        maximum_iou = _temporal_iou(predicted_xx, target_xx).max(dim=1).values
        targets[batch_index] = maximum_iou
        weights[batch_index] = (
            float(negative_candidate_weight)
            + (1.0 - float(negative_candidate_weight)) * maximum_iou
        ) * valid[batch_index].to(maximum_iou)

    raw = F.binary_cross_entropy_with_logits(
        quality_logits, targets.detach(), reduction="none"
    )
    return (raw * weights).sum() / weights.sum().clamp_min(1.0)


def flash_independent_zero_loss(
    zero_logits: torch.Tensor,
    exist_labels: torch.Tensor,
    positive_query_weight: float = 1.0,
) -> torch.Tensor:
    """Binary cross entropy for ``P(N=0)`` with optional positive reweighting."""
    logits = zero_logits.reshape(-1)
    exists = exist_labels.to(logits).reshape(-1)
    target_zero = 1.0 - exists
    raw = F.binary_cross_entropy_with_logits(logits, target_zero, reduction="none")
    weights = torch.where(
        exists > 0.5,
        torch.full_like(raw, float(positive_query_weight)),
        torch.ones_like(raw),
    )
    return (raw * weights).sum() / weights.sum().clamp_min(1.0)


def quality_calibrated_scores(
    foreground: torch.Tensor,
    quality_logits: torch.Tensor | None,
    alpha: float = 0.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Geometric foreground/quality calibration used for candidate ranking."""
    if quality_logits is None or float(alpha) <= 0:
        return foreground
    alpha = min(max(float(alpha), 0.0), 1.0)
    quality = torch.sigmoid(quality_logits)
    return (
        foreground.clamp_min(eps).pow(1.0 - alpha)
        * quality.clamp_min(eps).pow(alpha)
    )


__all__ = [
    "FlashCandidateQualityHead",
    "FlashIndependentZeroVerifier",
    "flash_candidate_quality_loss",
    "flash_independent_zero_loss",
    "quality_calibrated_scores",
]
