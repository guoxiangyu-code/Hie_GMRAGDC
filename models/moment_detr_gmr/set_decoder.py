"""Quality-aware and count-aware decoding for DETR moment queries."""

from __future__ import annotations

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
    if mode == "full":
        return list(ranking)
    if mode not in {"threshold", "adaptive", "hard"}:
        raise ValueError(f"Unknown decode mode: {mode}")

    # GREC-style dynamic set prediction: fixed top-k cannot express both null
    # and multi-target outputs, so retain every query above a score threshold.
    if mode == "threshold":
        return [
            index for index in ranking
            if float(scores[index]) >= float(window_score_threshold)
        ]
    if count_probabilities is None:
        return list(ranking)

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
