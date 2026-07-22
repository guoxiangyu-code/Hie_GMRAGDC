"""Shared training/evaluation runtime for isolated EaTR variants."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from eval.eval_main import evaluate_gmr

from .hierarchical_counter import hierarchical_count_probabilities
from .set_decoder import fuse_query_scores, hierarchical_two_stage_decode
from .spans import span_cxw_to_xx


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def write_jsonl(rows: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(value: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)


def move_inputs(model_inputs: dict[str, torch.Tensor], device: torch.device):
    return {key: value.to(device) for key, value in model_inputs.items()}


def _raw_query_fields(
    spans: torch.Tensor,
    scores: torch.Tensor,
    foreground: torch.Tensor,
    quality: torch.Tensor,
    ranking: list[int],
) -> dict[str, list[list[float]]]:
    """Serialize the ranking-aligned raw-query calibration contract."""
    query_count = int(scores.numel())
    expected_vector_shape = (query_count,)
    if tuple(spans.shape) != (query_count, 2):
        raise ValueError(
            f"raw-query spans must have shape ({query_count}, 2); "
            f"got {tuple(spans.shape)}"
        )
    for name, values in (("scores", scores), ("foreground", foreground), ("quality", quality)):
        if tuple(values.shape) != expected_vector_shape:
            raise ValueError(
                f"raw-query {name} must have shape {expected_vector_shape}; "
                f"got {tuple(values.shape)}"
            )
    if len(ranking) != query_count or sorted(ranking) != list(range(query_count)):
        raise ValueError("raw-query ranking must be a permutation of every query index")

    return {
        "all_query_windows": [
            [
                float(f"{float(spans[index, 0]):.4f}"),
                float(f"{float(spans[index, 1]):.4f}"),
                float(f"{float(scores[index]):.6f}"),
            ]
            for index in ranking
        ],
        "all_query_components": [
            [
                float(f"{float(foreground[index]):.6f}"),
                float(f"{float(quality[index]):.6f}"),
            ]
            for index in ranking
        ],
    }


@torch.no_grad()
def predict_views(
    model,
    loader,
    device: torch.device,
    *,
    modes: tuple[str, ...] = ("full",),
    max_predictions: int = 10,
    round_to_clip: bool = True,
    clip_length: float = 2.0,
    quality_alpha: float = 0.5,
    diversity_lambda: float = 0.0,
    existence_threshold: float = 0.4,
    count_confidence_threshold: float = 0.55,
    window_score_threshold: float = 0.1,
    save_raw_queries: bool = False,
):
    """Generate full-primary and optional count-adaptive views in one pass."""
    if not modes or any(mode not in {"full", "adaptive", "hard"} for mode in modes):
        raise ValueError(f"invalid decode modes: {modes}")
    model.eval()
    submissions = {mode: [] for mode in modes}
    for metadata, model_inputs, _ in loader:
        outputs = model(**move_inputs(model_inputs, device))
        foreground_scores = F.softmax(outputs["pred_logits"], dim=-1)[..., 0]
        scores = fuse_query_scores(
            foreground_scores,
            outputs.get("pred_quality_logits"),
            quality_alpha=quality_alpha,
        )
        spans = span_cxw_to_xx(outputs["pred_spans"]).clamp(0.0, 1.0)
        exist_scores = (
            torch.sigmoid(outputs["pred_exist_logits"])
            if "pred_exist_logits" in outputs else None
        )
        quality_scores = (
            torch.sigmoid(outputs["pred_quality_logits"])
            if "pred_quality_logits" in outputs else None
        )
        count_probabilities = (
            hierarchical_count_probabilities(outputs)
            if "pred_positive_count_logits" in outputs else None
        )

        for index, meta in enumerate(metadata):
            duration = float(meta["duration"])
            normalized_spans = spans[index].detach().cpu()
            sample_spans = normalized_spans * duration
            if round_to_clip:
                sample_spans = torch.round(sample_spans / clip_length) * clip_length
            sample_spans = sample_spans.clamp(0.0, duration)
            sample_scores = scores[index].detach().cpu()
            sample_foreground = foreground_scores[index].detach().cpu()
            sample_quality = (
                quality_scores[index].detach().cpu()
                if quality_scores is not None else torch.ones_like(sample_foreground)
            )
            sample_count = (
                count_probabilities[index].detach().cpu()
                if count_probabilities is not None else None
            )

            for mode in modes:
                decoded = hierarchical_two_stage_decode(
                    normalized_spans,
                    sample_scores,
                    sample_count,
                    mode=mode,
                    diversity_lambda=diversity_lambda,
                    existence_threshold=existence_threshold,
                    count_confidence_threshold=count_confidence_threshold,
                    window_score_threshold=window_score_threshold,
                )
                selected = decoded.selected[:max_predictions]
                if selected:
                    selected_tensor = torch.as_tensor(selected, dtype=torch.long)
                    windows = torch.cat([
                        sample_spans[selected_tensor],
                        sample_scores[selected_tensor, None],
                    ], dim=-1).tolist()
                else:
                    windows = []
                row = {
                    "qid": meta["qid"],
                    "query": meta["query"],
                    "vid": meta["vid"],
                    "pred_relevant_windows": [
                        [float(f"{start:.4f}"), float(f"{end:.4f}"), float(f"{score:.6f}")]
                        for start, end, score in windows
                    ],
                }
                if exist_scores is not None:
                    row["pred_exist_score"] = float(
                        f"{float(exist_scores[index].detach().cpu()):.6f}"
                    )
                if sample_count is not None:
                    row["pred_count"] = int(decoded.predicted_count)
                    row["pred_count_probs"] = [
                        float(f"{float(value):.6f}") for value in sample_count
                    ]
                if save_raw_queries:
                    row.update(_raw_query_fields(
                        sample_spans,
                        sample_scores,
                        sample_foreground,
                        sample_quality,
                        decoded.ranking,
                    ))
                submissions[mode].append(row)
    return submissions


def predict(model, loader, device: torch.device, *, decode_mode: str = "full", **kwargs):
    """Compatibility wrapper returning one requested decode view."""
    return predict_views(
        model, loader, device, modes=(decode_mode,), **kwargs
    )[decode_mode]


def official_metrics(submission: list[dict[str, Any]], ground_truth: list[dict[str, Any]],
                     *, map_num_workers: int = 1, verbose: bool = False):
    """Call the repository's official Soccer-GMR evaluator directly."""
    return evaluate_gmr(
        submission,
        ground_truth,
        k_list=(1, 3, 5),
        max_pred_windows=10,
        cls_thresholds=(0.4, 0.6, 0.8),
        gmiou_cls_threshold=0.4,
        map_num_workers=map_num_workers,
        verbose=verbose,
    )
