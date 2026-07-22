"""Quality-aware and count-aware decoding for DETR moment queries."""

from __future__ import annotations

from dataclasses import dataclass

import torch


def fuse_query_scores(
    foreground_scores: torch.Tensor,
    quality_logits: torch.Tensor | None,
    quality_alpha: float = 0.5,
) -> torch.Tensor:
    """Geometrically fuse foreground probability and predicted temporal IoU."""
    if quality_logits is None:
        return foreground_scores
    alpha = float(quality_alpha)
    quality = torch.sigmoid(quality_logits)
    eps = torch.finfo(foreground_scores.dtype).eps
    return foreground_scores.clamp_min(eps).pow(1.0 - alpha) \
        * quality.clamp_min(eps).pow(alpha)


def pairwise_temporal_iou(spans_xx: torch.Tensor) -> torch.Tensor:
    """Pairwise IoU for ``[start,end]`` spans."""
    starts = spans_xx[:, 0]
    ends = spans_xx[:, 1]
    intersection = (
        torch.minimum(ends[:, None], ends[None])
        - torch.maximum(starts[:, None], starts[None])
    ).clamp_min(0)
    lengths = (ends - starts).clamp_min(0)
    union = (lengths[:, None] + lengths[None] - intersection).clamp_min(1e-8)
    return intersection / union


def diversity_ranking(
    spans_xx: torch.Tensor,
    scores: torch.Tensor,
    diversity_lambda: float = 0.0,
) -> list[int]:
    """Greedy MMR ranking that penalizes overlap with already selected moments."""
    if scores.numel() == 0:
        return []
    if float(diversity_lambda) <= 0:
        return torch.argsort(scores, descending=True).tolist()

    pairwise_iou = pairwise_temporal_iou(spans_xx)
    remaining = set(range(scores.numel()))
    selected: list[int] = []
    eps = torch.finfo(scores.dtype).eps
    log_scores = scores.clamp_min(eps).log()
    while remaining:
        best_index = None
        best_utility = None
        for index in sorted(remaining):
            redundancy = (
                max(float(pairwise_iou[index, prior]) for prior in selected)
                if selected else 0.0
            )
            utility = float(log_scores[index]) - float(diversity_lambda) * redundancy
            if best_utility is None or utility > best_utility:
                best_index = index
                best_utility = utility
        assert best_index is not None
        selected.append(best_index)
        remaining.remove(best_index)
    return selected


def adaptive_count_indices(
    ranking: list[int],
    scores: torch.Tensor,
    count_probabilities: torch.Tensor | None,
    mode: str = "full",
    existence_threshold: float = 0.4,
    count_confidence_threshold: float = 0.55,
    window_score_threshold: float = 0.1,
) -> list[int]:
    """Apply HieA2G-style count selection without changing the query ranking."""
    if mode == "full" or count_probabilities is None:
        return list(ranking)
    if mode not in {"adaptive", "hard"}:
        raise ValueError(f"Unknown decode mode: {mode}")

    # Factorized inference: decide existence first, then cardinality given a
    # positive query. Taking argmax over the five joint probabilities would
    # recreate flat-classifier null collapse whenever the conditional count is
    # uncertain (e.g. P(exists)=.6 with four near-uniform positive classes).
    existence_probability = 1.0 - float(count_probabilities[0])
    if existence_probability <= float(existence_threshold):
        return []

    conditional = count_probabilities[1:] / count_probabilities[1:].sum().clamp_min(1e-8)
    predicted_class = int(torch.argmax(conditional).item()) + 1
    conditional_confidence = float(conditional.max())
    if predicted_class < 4 and (
        mode == "hard" or conditional_confidence >= float(count_confidence_threshold)
    ):
        return list(ranking[:predicted_class])

    # The 4+ class and uncertain positive counts use the paper's threshold
    # fallback, with a minimum of four for a confident 4+ prediction.
    retained = [index for index in ranking if float(scores[index]) >= float(window_score_threshold)]
    if predicted_class == 4 and len(retained) < min(4, len(ranking)):
        retained = list(ranking[: min(4, len(ranking))])
    if not retained and ranking:
        retained = [ranking[0]]
    return retained


def factorized_count_prediction(
    count_probabilities: torch.Tensor,
    existence_threshold: float = 0.4,
) -> int:
    """Decode ``0`` first, then the positive-conditional count class."""
    existence_probability = 1.0 - float(count_probabilities[0])
    if existence_probability <= float(existence_threshold):
        return 0
    conditional = count_probabilities[1:]
    conditional = conditional / conditional.sum().clamp_min(1e-8)
    return int(torch.argmax(conditional).item()) + 1


@dataclass(frozen=True)
class TwoStageDecode:
    """Auditable output of ranking followed by hierarchical count selection."""

    ranking: list[int]
    selected: list[int]
    predicted_count: int | None


def hierarchical_two_stage_decode(
    spans_xx: torch.Tensor,
    scores: torch.Tensor,
    count_probabilities: torch.Tensor | None,
    *,
    mode: str = "full",
    diversity_lambda: float = 0.0,
    existence_threshold: float = 0.4,
    count_confidence_threshold: float = 0.55,
    window_score_threshold: float = 0.1,
) -> TwoStageDecode:
    """Stage 1 ranks quality-aware queries; stage 2 selects cardinality."""
    ranking = diversity_ranking(
        spans_xx, scores, diversity_lambda=diversity_lambda
    )
    selected = adaptive_count_indices(
        ranking,
        scores,
        count_probabilities,
        mode=mode,
        existence_threshold=existence_threshold,
        count_confidence_threshold=count_confidence_threshold,
        window_score_threshold=window_score_threshold,
    )
    predicted_count = (
        factorized_count_prediction(count_probabilities, existence_threshold)
        if count_probabilities is not None else None
    )
    return TwoStageDecode(
        ranking=list(ranking), selected=list(selected), predicted_count=predicted_count
    )
