"""Standalone one-dimensional temporal de-duplication utilities.

The functions in this module are inference-only selection helpers.  They do not
depend on a model, a trainer, or project-specific configuration.  All methods
operate within one video/query pair and return indices into the original input.

Geometry alone cannot tell apart two semantically distinct events with exactly
the same boundaries.  Callers that have such information can pass
``protected_pairs``; protected pairs are never clustered, suppressed, or
score-decayed.  ``protect_center_distance`` supplies a weaker geometry-only
fallback for heavily overlapping events whose centres are measurably different.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch


SUPPORTED_METHODS = (
    "none",
    "direct_topk",
    "hard_nms",
    "gaussian_soft_nms",
    "diou_nms",
    "cluster_vote_soft_nms",
)


def _empty_result(
    spans: torch.Tensor, scores: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        spans.new_empty((0, 2)),
        scores.new_empty((0,)),
        torch.empty((0,), dtype=torch.long, device=spans.device),
    )


def _validate_inputs(
    spans: torch.Tensor,
    scores: torch.Tensor,
    protected_pairs: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    if not isinstance(spans, torch.Tensor) or not isinstance(scores, torch.Tensor):
        raise TypeError("spans and scores must be torch tensors")
    if spans.ndim != 2 or spans.shape[1] != 2:
        raise ValueError("spans must have shape [N, 2]")
    if scores.ndim != 1 or scores.shape[0] != spans.shape[0]:
        raise ValueError("scores must have shape [N]")
    if not spans.is_floating_point() or not scores.is_floating_point():
        raise TypeError("spans and scores must use floating-point dtypes")
    if spans.device != scores.device:
        raise ValueError("spans and scores must be on the same device")
    if spans.numel() and not bool(torch.isfinite(spans).all()):
        raise ValueError("spans must contain only finite values")
    if scores.numel() and not bool(torch.isfinite(scores).all()):
        raise ValueError("scores must contain only finite values")
    if scores.numel() and bool((scores < 0).any()):
        raise ValueError("scores must be non-negative confidence values")
    if spans.numel() and bool((spans[:, 1] <= spans[:, 0]).any()):
        raise ValueError("every span must have positive duration (end > start)")

    if protected_pairs is None:
        return None
    if not isinstance(protected_pairs, torch.Tensor):
        raise TypeError("protected_pairs must be a torch tensor")
    expected_shape = (spans.shape[0], spans.shape[0])
    if tuple(protected_pairs.shape) != expected_shape:
        raise ValueError(
            "protected_pairs must have shape [N, N], got "
            f"{tuple(protected_pairs.shape)}"
        )
    protected = protected_pairs.to(device=spans.device, dtype=torch.bool)
    # Pairwise distinctness is symmetric even if a caller supplied one triangle.
    protected = protected | protected.transpose(0, 1)
    protected = protected.clone()
    protected.fill_diagonal_(False)
    return protected


def _validate_parameters(
    *,
    method: str,
    iou_threshold: float,
    diou_threshold: float,
    soft_sigma: float,
    soft_iou_threshold: float,
    cluster_iou_threshold: float,
    cluster_center_distance_threshold: float,
    cluster_duration_ratio_threshold: float,
    score_threshold: float,
    max_output: Optional[int],
    voting_score_power: float,
    voting_iou_power: float,
    protect_center_distance: Optional[float],
    eps: float,
) -> None:
    if method not in SUPPORTED_METHODS:
        raise ValueError(
            f"unknown temporal de-duplication method {method!r}; "
            f"expected one of {SUPPORTED_METHODS}"
        )
    for name, value in (
        ("iou_threshold", iou_threshold),
        ("soft_iou_threshold", soft_iou_threshold),
        ("cluster_iou_threshold", cluster_iou_threshold),
        ("cluster_duration_ratio_threshold", cluster_duration_ratio_threshold),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
    if not -1.0 <= diou_threshold <= 1.0:
        raise ValueError("diou_threshold must be in [-1, 1]")
    if soft_sigma <= 0.0:
        raise ValueError("soft_sigma must be positive")
    if cluster_center_distance_threshold < 0.0:
        raise ValueError("cluster_center_distance_threshold must be non-negative")
    if score_threshold < 0.0:
        raise ValueError("score_threshold must be non-negative")
    if max_output is not None and (not isinstance(max_output, int) or max_output < 0):
        raise ValueError("max_output must be a non-negative integer or None")
    if voting_score_power < 0.0 or voting_iou_power < 0.0:
        raise ValueError("voting powers must be non-negative")
    if protect_center_distance is not None and protect_center_distance < 0.0:
        raise ValueError("protect_center_distance must be non-negative or None")
    if eps <= 0.0:
        raise ValueError("eps must be positive")


def _indices_tensor(indices, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(indices, dtype=torch.long, device=device)


def _ranked_indices(scores: torch.Tensor, score_threshold: float) -> list[int]:
    valid = torch.nonzero(scores >= score_threshold, as_tuple=False).flatten()
    if valid.numel() == 0:
        return []
    ranked = valid[torch.argsort(scores[valid], descending=True, stable=True)]
    return [int(index) for index in ranked]


def _temporal_iou(anchor: torch.Tensor, spans: torch.Tensor, eps: float) -> torch.Tensor:
    if spans.numel() == 0:
        return spans.new_empty((0,))
    left = torch.maximum(anchor[0], spans[:, 0])
    right = torch.minimum(anchor[1], spans[:, 1])
    intersection = (right - left).clamp_min(0.0)
    anchor_duration = anchor[1] - anchor[0]
    durations = spans[:, 1] - spans[:, 0]
    union = anchor_duration + durations - intersection
    return intersection / union.clamp_min(eps)


def _normalized_center_distance(
    anchor: torch.Tensor, spans: torch.Tensor, eps: float
) -> torch.Tensor:
    """Centre distance normalized by the shorter duration.

    The shorter duration is deliberately used here: it prevents a long proposal
    from making two visibly shifted short events appear artificially close.
    """

    anchor_duration = anchor[1] - anchor[0]
    durations = spans[:, 1] - spans[:, 0]
    anchor_center = 0.5 * (anchor[0] + anchor[1])
    centres = 0.5 * (spans[:, 0] + spans[:, 1])
    denominator = torch.minimum(anchor_duration, durations).clamp_min(eps)
    return (centres - anchor_center).abs() / denominator


def _duration_ratio(anchor: torch.Tensor, spans: torch.Tensor, eps: float) -> torch.Tensor:
    anchor_duration = anchor[1] - anchor[0]
    durations = spans[:, 1] - spans[:, 0]
    return torch.minimum(anchor_duration, durations) / torch.maximum(
        anchor_duration, durations
    ).clamp_min(eps)


def _pair_protection(
    anchor_index: int,
    other_indices: torch.Tensor,
    spans: torch.Tensor,
    protected_pairs: Optional[torch.Tensor],
    protect_center_distance: Optional[float],
    eps: float,
) -> torch.Tensor:
    result = torch.zeros(
        other_indices.shape[0], dtype=torch.bool, device=spans.device
    )
    if protected_pairs is not None:
        result |= protected_pairs[anchor_index, other_indices]
    if protect_center_distance is not None and other_indices.numel():
        result |= _normalized_center_distance(
            spans[anchor_index], spans[other_indices], eps
        ) >= protect_center_distance
    return result


def _finalize_original(
    spans: torch.Tensor,
    scores: torch.Tensor,
    indices: list[int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not indices:
        return _empty_result(spans, scores)
    index_tensor = _indices_tensor(indices, spans.device)
    return spans[index_tensor].clone(), scores[index_tensor].clone(), index_tensor


def _hard_nms(
    spans: torch.Tensor,
    scores: torch.Tensor,
    *,
    threshold: float,
    score_threshold: float,
    max_output: Optional[int],
    protected_pairs: Optional[torch.Tensor],
    protect_center_distance: Optional[float],
    use_diou: bool,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    remaining = _ranked_indices(scores, score_threshold)
    selected: list[int] = []

    while remaining and (max_output is None or len(selected) < max_output):
        anchor_index = remaining.pop(0)
        selected.append(anchor_index)
        if not remaining:
            continue
        other_indices = _indices_tensor(remaining, spans.device)
        overlap = _temporal_iou(spans[anchor_index], spans[other_indices], eps)
        if use_diou:
            anchor_center = 0.5 * (spans[anchor_index, 0] + spans[anchor_index, 1])
            centres = 0.5 * (
                spans[other_indices, 0] + spans[other_indices, 1]
            )
            enclosing_left = torch.minimum(
                spans[anchor_index, 0], spans[other_indices, 0]
            )
            enclosing_right = torch.maximum(
                spans[anchor_index, 1], spans[other_indices, 1]
            )
            enclosing_length = (enclosing_right - enclosing_left).clamp_min(eps)
            overlap = overlap - ((centres - anchor_center) / enclosing_length).square()

        protected = _pair_protection(
            anchor_index,
            other_indices,
            spans,
            protected_pairs,
            protect_center_distance,
            eps,
        )
        suppress = (overlap >= threshold) & ~protected
        remaining = [
            index
            for index, is_suppressed in zip(remaining, suppress.tolist())
            if not is_suppressed
        ]

    return _finalize_original(spans, scores, selected)


def _gaussian_soft_nms(
    spans: torch.Tensor,
    scores: torch.Tensor,
    *,
    sigma: float,
    iou_threshold: float,
    score_threshold: float,
    max_output: Optional[int],
    protected_pairs: Optional[torch.Tensor],
    protect_center_distance: Optional[float],
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    updated_scores = scores.clone()
    remaining = _ranked_indices(updated_scores, score_threshold)
    selected: list[int] = []
    selected_scores = []

    while remaining and (max_output is None or len(selected) < max_output):
        remaining_tensor = _indices_tensor(remaining, spans.device)
        best_position = int(torch.argmax(updated_scores[remaining_tensor]))
        anchor_index = remaining.pop(best_position)
        anchor_score = updated_scores[anchor_index]
        if float(anchor_score) < score_threshold:
            break
        selected.append(anchor_index)
        selected_scores.append(anchor_score.clone())
        if not remaining:
            continue

        other_indices = _indices_tensor(remaining, spans.device)
        overlap = _temporal_iou(spans[anchor_index], spans[other_indices], eps)
        protected = _pair_protection(
            anchor_index,
            other_indices,
            spans,
            protected_pairs,
            protect_center_distance,
            eps,
        )
        should_decay = (overlap >= iou_threshold) & ~protected
        decay = torch.ones_like(overlap)
        decay[should_decay] = torch.exp(-overlap[should_decay].square() / sigma)
        updated_scores[other_indices] = updated_scores[other_indices] * decay
        remaining = [
            index
            for index in remaining
            if float(updated_scores[index]) >= score_threshold
        ]

    if not selected:
        return _empty_result(spans, scores)
    selected_indices = _indices_tensor(selected, spans.device)
    return (
        spans[selected_indices].clone(),
        torch.stack(selected_scores),
        selected_indices,
    )


def _cluster_vote_soft_nms(
    spans: torch.Tensor,
    scores: torch.Tensor,
    *,
    soft_sigma: float,
    soft_iou_threshold: float,
    cluster_iou_threshold: float,
    cluster_center_distance_threshold: float,
    cluster_duration_ratio_threshold: float,
    score_threshold: float,
    max_output: Optional[int],
    voting_score_power: float,
    voting_iou_power: float,
    protected_pairs: Optional[torch.Tensor],
    protect_center_distance: Optional[float],
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    remaining = _ranked_indices(scores, score_threshold)
    fused_spans = []
    fused_scores = []
    representatives: list[int] = []
    clusters: list[list[int]] = []

    while remaining:
        anchor_index = remaining.pop(0)
        cluster = [anchor_index]
        if remaining:
            other_indices = _indices_tensor(remaining, spans.device)
            overlap = _temporal_iou(spans[anchor_index], spans[other_indices], eps)
            centre_distance = _normalized_center_distance(
                spans[anchor_index], spans[other_indices], eps
            )
            duration_ratio = _duration_ratio(
                spans[anchor_index], spans[other_indices], eps
            )
            protected = _pair_protection(
                anchor_index,
                other_indices,
                spans,
                protected_pairs,
                protect_center_distance,
                eps,
            )
            is_duplicate = (
                (overlap >= cluster_iou_threshold)
                & (centre_distance <= cluster_center_distance_threshold)
                & (duration_ratio >= cluster_duration_ratio_threshold)
                & ~protected
            )
            duplicate_flags = is_duplicate.tolist()
            cluster.extend(
                index
                for index, duplicate in zip(remaining, duplicate_flags)
                if duplicate
            )
            remaining = [
                index
                for index, duplicate in zip(remaining, duplicate_flags)
                if not duplicate
            ]

        member_indices = _indices_tensor(cluster, spans.device)
        member_overlap = _temporal_iou(
            spans[anchor_index], spans[member_indices], eps
        )
        weights = scores[member_indices].clamp_min(eps).pow(voting_score_power)
        weights = weights * member_overlap.clamp_min(eps).pow(voting_iou_power)
        weights = weights / weights.sum().clamp_min(eps)
        fused_spans.append((weights[:, None] * spans[member_indices]).sum(dim=0))
        # Max aggregation avoids rewarding a model merely for emitting duplicates.
        fused_scores.append(scores[member_indices].max())
        representatives.append(anchor_index)
        clusters.append(cluster)

    if not fused_spans:
        return _empty_result(spans, scores)

    clustered_spans = torch.stack(fused_spans)
    clustered_scores = torch.stack(fused_scores)
    cluster_count = len(clusters)
    clustered_protection = torch.zeros(
        (cluster_count, cluster_count), dtype=torch.bool, device=spans.device
    )
    if protected_pairs is not None:
        for first in range(cluster_count):
            first_indices = _indices_tensor(clusters[first], spans.device)
            for second in range(first + 1, cluster_count):
                second_indices = _indices_tensor(clusters[second], spans.device)
                if bool(
                    protected_pairs[first_indices[:, None], second_indices[None, :]].any()
                ):
                    clustered_protection[first, second] = True
                    clustered_protection[second, first] = True

    selected_spans, selected_scores, cluster_indices = _gaussian_soft_nms(
        clustered_spans,
        clustered_scores,
        sigma=soft_sigma,
        iou_threshold=soft_iou_threshold,
        score_threshold=score_threshold,
        max_output=max_output,
        protected_pairs=clustered_protection,
        protect_center_distance=protect_center_distance,
        eps=eps,
    )
    representative_tensor = _indices_tensor(representatives, spans.device)
    return selected_spans, selected_scores, representative_tensor[cluster_indices]


def temporal_deduplicate(
    spans: torch.Tensor,
    scores: torch.Tensor,
    *,
    method: str = "cluster_vote_soft_nms",
    iou_threshold: float = 0.5,
    diou_threshold: float = 0.5,
    soft_sigma: float = 2.0,
    soft_iou_threshold: float = 0.0,
    cluster_iou_threshold: float = 0.9,
    cluster_center_distance_threshold: float = 0.1,
    cluster_duration_ratio_threshold: float = 0.8,
    score_threshold: float = 0.0,
    max_output: Optional[int] = None,
    voting_score_power: float = 1.0,
    voting_iou_power: float = 1.0,
    protect_center_distance: Optional[float] = None,
    protected_pairs: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """De-duplicate and rank temporal spans.

    Args:
        spans: Floating tensor ``[N, 2]`` in ``[start, end]`` format.
        scores: Non-negative confidence tensor ``[N]``.
        method: One of :data:`SUPPORTED_METHODS`.
        iou_threshold: Suppression threshold for ``hard_nms``.
        diou_threshold: DIoU-similarity threshold for ``diou_nms``.
        soft_sigma: Gaussian Soft-NMS variance control; larger is gentler.
        soft_iou_threshold: Minimum tIoU at which Gaussian decay is applied.
        cluster_iou_threshold: Minimum tIoU for boundary-voting clusters.
        cluster_center_distance_threshold: Maximum centre distance, normalized
            by the shorter duration, for boundary-voting clusters.
        cluster_duration_ratio_threshold: Minimum shorter/longer duration ratio
            for boundary-voting clusters.
        score_threshold: Inclusive minimum score, applied before and after decay.
        max_output: Optional maximum number of returned event hypotheses.
        voting_score_power: Confidence exponent in boundary voting.
        voting_iou_power: Anchor-tIoU exponent in boundary voting.
        protect_center_distance: If set, pairs whose centre distance divided by
            shorter duration is at least this value are protected from all
            suppression, decay, and clustering.
        protected_pairs: Optional boolean-like ``[N, N]`` tensor.  A true entry
            explicitly marks two predictions as distinct events.  The mask is
            symmetrized and its diagonal is ignored.
        eps: Numerical stability constant.

    Returns:
        ``(selected_spans, selected_scores, original_indices)``.  Outputs are
        ordered by their final descending scores.  For a voted span, its index
        is the highest-scoring original member that anchored the cluster.

    Notes:
        ``none`` and its explicit experiment alias ``direct_topk`` perform only
        common score filtering, ranking, and optional truncation.  They leave
        coordinates and scores unchanged.  For a fixed-K comparison, pass the
        same ``max_output=K`` to ``direct_topk`` and the chosen de-duplication
        method.
    """

    protected_pairs = _validate_inputs(spans, scores, protected_pairs)
    _validate_parameters(
        method=method,
        iou_threshold=iou_threshold,
        diou_threshold=diou_threshold,
        soft_sigma=soft_sigma,
        soft_iou_threshold=soft_iou_threshold,
        cluster_iou_threshold=cluster_iou_threshold,
        cluster_center_distance_threshold=cluster_center_distance_threshold,
        cluster_duration_ratio_threshold=cluster_duration_ratio_threshold,
        score_threshold=score_threshold,
        max_output=max_output,
        voting_score_power=voting_score_power,
        voting_iou_power=voting_iou_power,
        protect_center_distance=protect_center_distance,
        eps=eps,
    )
    if spans.shape[0] == 0 or max_output == 0:
        return _empty_result(spans, scores)

    if method in ("none", "direct_topk"):
        selected = _ranked_indices(scores, score_threshold)
        if max_output is not None:
            selected = selected[:max_output]
        return _finalize_original(spans, scores, selected)
    if method == "hard_nms":
        return _hard_nms(
            spans,
            scores,
            threshold=iou_threshold,
            score_threshold=score_threshold,
            max_output=max_output,
            protected_pairs=protected_pairs,
            protect_center_distance=protect_center_distance,
            use_diou=False,
            eps=eps,
        )
    if method == "diou_nms":
        return _hard_nms(
            spans,
            scores,
            threshold=diou_threshold,
            score_threshold=score_threshold,
            max_output=max_output,
            protected_pairs=protected_pairs,
            protect_center_distance=protect_center_distance,
            use_diou=True,
            eps=eps,
        )
    if method == "gaussian_soft_nms":
        return _gaussian_soft_nms(
            spans,
            scores,
            sigma=soft_sigma,
            iou_threshold=soft_iou_threshold,
            score_threshold=score_threshold,
            max_output=max_output,
            protected_pairs=protected_pairs,
            protect_center_distance=protect_center_distance,
            eps=eps,
        )
    return _cluster_vote_soft_nms(
        spans,
        scores,
        soft_sigma=soft_sigma,
        soft_iou_threshold=soft_iou_threshold,
        cluster_iou_threshold=cluster_iou_threshold,
        cluster_center_distance_threshold=cluster_center_distance_threshold,
        cluster_duration_ratio_threshold=cluster_duration_ratio_threshold,
        score_threshold=score_threshold,
        max_output=max_output,
        voting_score_power=voting_score_power,
        voting_iou_power=voting_iou_power,
        protected_pairs=protected_pairs,
        protect_center_distance=protect_center_distance,
        eps=eps,
    )


__all__ = ["SUPPORTED_METHODS", "temporal_deduplicate"]
