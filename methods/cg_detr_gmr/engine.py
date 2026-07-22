"""Training/evaluation primitives for the Soccer-GMR CG-DETR adapter."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from eval.eval_main import evaluate_gmr

from .dataset import SoccerGMRDataset, collate_fn, prepare_batch
from .model import build_model
from .span_utils import span_cxw_to_xx
from models.moment_detr_gmr.hierarchical_counter import hierarchical_count_probabilities
from models.moment_detr_gmr.set_decoder import (
    adaptive_count_indices,
    diversity_ranking,
    fuse_query_scores,
)


RAW_QUERY_SCHEMA_VERSION = 2


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(value: str) -> torch.device:
    if value.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(value)


def build_components(args):
    args.device = str(resolve_device(str(args.device)))
    model, criterion = build_model(args)
    device = torch.device(args.device)
    return model.to(device), criterion.to(device), device


def make_dataset(
    args,
    annotation_path: str,
    *,
    sample_mode: str = "mixed",
    max_samples: int | None = None,
) -> SoccerGMRDataset:
    return SoccerGMRDataset(
        annotation_path,
        args.video_feature_dirs,
        args.text_feature_dir,
        max_q_l=args.max_q_l,
        max_v_l=args.max_v_l,
        max_windows=args.max_windows,
        clip_length=args.clip_length,
        use_tef=args.use_tef,
        trim_text_by_attention_mask=args.trim_text_by_attention_mask,
        sample_mode=sample_mode,
        max_samples=max_samples,
    )


def make_loader(dataset, *, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )


def weighted_loss(loss_dict: dict, weight_dict: dict) -> torch.Tensor:
    terms = [loss_dict[key] * weight_dict[key] for key in loss_dict if key in weight_dict]
    if not terms:
        raise RuntimeError("criterion produced no weighted losses")
    return sum(terms)


def train_one_epoch(model, criterion, loader, optimizer, device, *, grad_clip: float) -> dict[str, float]:
    model.train()
    criterion.train()
    totals: dict[str, float] = {}
    examples = 0
    for batch in loader:
        metadata, inputs, targets = prepare_batch(batch, device)
        outputs = model(**inputs, targets=targets)
        loss_dict = criterion(outputs, targets)
        loss = weighted_loss(loss_dict, criterion.weight_dict)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss: {loss_dict}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        batch_size = len(metadata)
        examples += batch_size
        for key, value in {**loss_dict, "loss_overall": loss}.items():
            scalar = float(value.detach()) if torch.is_tensor(value) else float(value)
            totals[key] = totals.get(key, 0.0) + scalar * batch_size
    return {key: value / max(examples, 1) for key, value in totals.items()}


def _round_windows(windows: torch.Tensor, clip_length: float) -> torch.Tensor:
    return torch.round(windows / clip_length) * clip_length


def _raw_query_replay_record(
    normalized_spans: torch.Tensor,
    foreground: torch.Tensor,
    quality: torch.Tensor,
    *,
    duration: float,
    quality_available: bool,
    existence_score: torch.Tensor | None,
    count_probabilities: torch.Tensor | None,
) -> dict:
    """Serialize full-precision model outputs in query-index order."""

    query_count = int(foreground.numel())
    if tuple(normalized_spans.shape) != (query_count, 2):
        raise ValueError("raw replay spans must have shape [num_queries,2]")
    if tuple(quality.shape) != (query_count,):
        raise ValueError("raw replay quality must have shape [num_queries]")
    record = {
        "schema_version": RAW_QUERY_SCHEMA_VERSION,
        "ordering": "query_index",
        "span_units": "normalized",
        "duration_seconds": float(duration),
        "quality_available": bool(quality_available),
        "queries": [
            {
                "query_index": index,
                "span": [
                    float(normalized_spans[index, 0]),
                    float(normalized_spans[index, 1]),
                ],
                "foreground": float(foreground[index]),
                "quality": float(quality[index]),
            }
            for index in range(query_count)
        ],
        "submission_format": {
            "span_decimals": 4,
            "score_decimals": 4,
            "existence_decimals": 4,
            "count_probability_decimals": 6,
        },
    }
    if existence_score is not None:
        record["existence_score"] = float(existence_score)
    if count_probabilities is not None:
        record["count_probabilities"] = [
            float(value) for value in count_probabilities
        ]
    return record


@torch.no_grad()
def predict(
    model,
    loader,
    device,
    *,
    clip_length: float,
    round_to_clip: bool = True,
    quality_alpha: float = 0.5,
    diversity_lambda: float = 0.0,
    decode_mode: str = "full",
    existence_threshold: float = 0.4,
    count_confidence_threshold: float = 0.55,
    window_score_threshold: float = 0.1,
    save_raw_queries: bool = False,
) -> list[dict]:
    model.eval()
    submission: list[dict] = []
    for batch in loader:
        metadata, inputs, _ = prepare_batch(batch, device)
        outputs = model(**inputs)
        foreground = F.softmax(outputs["pred_logits"], dim=-1)[..., 0]
        scores = fuse_query_scores(
            foreground,
            outputs.get("pred_quality_logits"),
            quality_alpha=quality_alpha,
        ).cpu()
        foreground = foreground.cpu()
        quality_available = "pred_quality_logits" in outputs
        quality = (
            torch.sigmoid(outputs["pred_quality_logits"]).cpu()
            if quality_available else torch.ones_like(foreground)
        )
        spans = span_cxw_to_xx(outputs["pred_spans"].cpu()).clamp(0, 1)
        existence = (
            torch.sigmoid(outputs["pred_exist_logits"]).cpu()
            if "pred_exist_logits" in outputs else None
        )
        count_probabilities = (
            hierarchical_count_probabilities(outputs).cpu()
            if "pred_positive_count_logits" in outputs else None
        )
        for row_index, meta in enumerate(metadata):
            seconds = spans[row_index] * float(meta["duration"])
            seconds[:, 0].clamp_(0, float(meta["duration"]))
            seconds[:, 1].clamp_(0, float(meta["duration"]))
            if round_to_clip:
                seconds = _round_windows(seconds, clip_length)
                seconds.clamp_(0, float(meta["duration"]))
            ranked = diversity_ranking(
                spans[row_index], scores[row_index], diversity_lambda=diversity_lambda
            )
            selected = adaptive_count_indices(
                ranked,
                scores[row_index],
                count_probabilities[row_index] if count_probabilities is not None else None,
                mode=decode_mode,
                existence_threshold=existence_threshold,
                count_confidence_threshold=count_confidence_threshold,
                window_score_threshold=window_score_threshold,
            )
            windows = torch.cat([seconds, scores[row_index, :, None]], dim=-1)
            result = {
                "qid": meta["qid"],
                "query": meta.get("query", ""),
                "vid": meta["vid"],
                "pred_relevant_windows": [
                    [float(f"{float(value):.4f}") for value in windows[index]]
                    for index in selected
                ],
            }
            if existence is not None:
                result["pred_exist_score"] = float(f"{float(existence[row_index]):.4f}")
            if count_probabilities is not None:
                sample_count = count_probabilities[row_index]
                predicted_exists = 1.0 - float(sample_count[0]) > existence_threshold
                positive_count = int(torch.argmax(sample_count[1:]).item()) + 1
                result["pred_count"] = positive_count if predicted_exists else 0
                result["pred_count_probs"] = [
                    float(f"{float(value):.6f}")
                    for value in sample_count
                ]
            if save_raw_queries:
                result["raw_query_replay"] = _raw_query_replay_record(
                    spans[row_index],
                    foreground[row_index],
                    quality[row_index],
                    duration=float(meta["duration"]),
                    quality_available=quality_available,
                    existence_score=(
                        existence[row_index] if existence is not None else None
                    ),
                    count_probabilities=(
                        count_probabilities[row_index]
                        if count_probabilities is not None else None
                    ),
                )
                # Legacy fields remain for old artifacts.  The v2 decoder uses
                # only raw_query_replay and recomputes ranking from query indices.
                result["all_query_windows"] = [
                    [
                        float(f"{float(seconds[index, 0]):.4f}"),
                        float(f"{float(seconds[index, 1]):.4f}"),
                        float(f"{float(scores[row_index, index]):.6f}"),
                    ]
                    for index in ranked
                ]
                result["all_query_components"] = [
                    [
                        float(f"{float(foreground[row_index, index]):.6f}"),
                        float(f"{float(quality[row_index, index]):.6f}"),
                    ]
                    for index in ranked
                ]
            submission.append(result)
    return submission


def evaluate_submission(
    submission: list[dict],
    ground_truth: list[dict],
    *,
    gmiou_threshold: float = 0.4,
    map_num_workers: int = 1,
) -> dict:
    return evaluate_gmr(
        submission,
        ground_truth,
        k_list=(1, 3, 5),
        max_pred_windows=10,
        cls_thresholds=(0.4, 0.6, 0.8),
        gmiou_cls_threshold=gmiou_threshold,
        map_num_workers=map_num_workers,
        verbose=False,
    )


def save_jsonl(rows: Iterable[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_json(value: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def load_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def harmonic_joint(map_score: float, gmiou_score: float) -> float:
    if map_score <= 0 or gmiou_score <= 0:
        return 0.0
    return 2.0 * map_score * gmiou_score / (map_score + gmiou_score)
