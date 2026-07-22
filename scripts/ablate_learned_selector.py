#!/usr/bin/env python3
"""Validation-only ablation for learned same-event and soft-count selection."""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.eval_main import evaluate_gmr
from eval.normalization import normalize_ground_truth
from eval.utils import load_jsonl
from models.moment_detr_gmr.learned_selector import (
    cautious_complete_link_fusion,
    learned_mmr_select,
)
from scripts.ablate_temporal_dedup import assert_validation_input_path


def _write_json(path: Path, value) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def _write_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _row_tensors(row: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    qid = row.get("qid")
    windows = torch.as_tensor(row.get("all_query_windows"), dtype=torch.float32)
    duplicate = torch.as_tensor(row.get("pred_same_event_probs"), dtype=torch.float32)
    count = torch.as_tensor(row.get("pred_count_probs"), dtype=torch.float32)
    if windows.ndim != 2 or windows.shape[1] != 3 or windows.shape[0] == 0:
        raise ValueError(f"qid={qid!r}: all_query_windows must be non-empty [Q,3]")
    if duplicate.shape != (windows.shape[0], windows.shape[0]):
        raise ValueError(f"qid={qid!r}: pred_same_event_probs must be [Q,Q]")
    query_indices = row.get("all_query_indices")
    if (
        not isinstance(query_indices, list)
        or sorted(query_indices) != list(range(windows.shape[0]))
    ):
        raise ValueError(
            f"qid={qid!r}: all_query_indices must bind ranked windows to pairwise rows"
        )
    if count.shape != (5,) or not torch.isclose(count.sum(), torch.tensor(1.0), atol=1e-4):
        raise ValueError(f"qid={qid!r}: pred_count_probs must be normalized [5]")
    if not all(torch.isfinite(value).all() for value in (windows, duplicate, count)):
        raise ValueError(f"qid={qid!r}: learned-selector fields contain non-finite values")
    return windows[:, :2], windows[:, 2], duplicate, count


def decode(rows: list[dict], config: dict) -> list[dict]:
    output = []
    for source in rows:
        spans, scores, duplicate, count = _row_tensors(source)
        mode = config["mode"]
        if mode == "direct_topk":
            selected = torch.argsort(scores, descending=True)[: config["k"]].tolist()
            final_spans, final_scores = spans[selected], scores[selected]
        else:
            fixed = mode == "learned_topk"
            selection = learned_mmr_select(
                scores,
                duplicate,
                max_output=config["k"] if fixed else config["max_output"],
                redundancy_lambda=config["redundancy_lambda"],
                count_probabilities=None if fixed else count,
                count_prior_weight=0.0 if fixed else config["count_prior_weight"],
                stop_threshold=float("-inf") if fixed else config["stop_threshold"],
            )
            selected = selection.selected
            if mode == "learned_soft_count_fusion":
                final_spans, final_scores = cautious_complete_link_fusion(
                    spans,
                    scores,
                    duplicate,
                    selected,
                    same_event_threshold=config["same_event_threshold"],
                    boundary_std_threshold=config["boundary_std_threshold"],
                )
            else:
                final_spans, final_scores = spans[selected], scores[selected]
        row = copy.deepcopy(source)
        row["pred_relevant_windows"] = [
            [
                float(f"{float(span[0]):.4f}"),
                float(f"{float(span[1]):.4f}"),
                float(f"{float(score):.4f}"),
            ]
            for span, score in zip(final_spans, final_scores)
        ]
        row["selected_count"] = len(row["pred_relevant_windows"])
        output.append(row)
    return output


def _evaluate(submission, ground_truth, workers: int) -> dict:
    return evaluate_gmr(
        submission,
        ground_truth,
        k_list=(1, 3, 5),
        max_pred_windows=10,
        cls_thresholds=(0.4, 0.6, 0.8),
        gmiou_cls_threshold=0.4,
        map_num_workers=workers,
        verbose=False,
    )


def _record(config, submission, ground_truth, workers, reference) -> dict:
    metrics = _evaluate(submission, ground_truth, workers)
    brief = dict(metrics["brief"])
    score = min(
        float(brief["mAP"]) / max(float(reference["mAP"]), 1e-8),
        float(brief["G-mIoU@3"]) / max(float(reference["G-mIoU@3"]), 1e-8),
    )
    return {
        "config": config,
        "selection_score": score,
        "mean_selected_count": sum(
            len(row["pred_relevant_windows"]) for row in submission
        ) / max(len(submission), 1),
        "brief": brief,
    }


def run(args: argparse.Namespace) -> dict:
    prediction_path = assert_validation_input_path(args.prediction_path, role="prediction")
    gt_path = assert_validation_input_path(args.gt_path, role="ground truth")
    rows = load_jsonl(str(prediction_path))
    gt_raw = load_jsonl(str(gt_path))
    ground_truth, _ = normalize_ground_truth(gt_raw, None, drop_empty_gt=False)
    if {row["qid"] for row in rows} != {row["qid"] for row in ground_truth}:
        raise ValueError("Prediction/validation ground-truth qid coverage differs")
    for row in rows:
        _row_tensors(row)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    direct_config = {"mode": "direct_topk", "k": args.fixed_k}
    direct_submission = decode(rows, direct_config)
    direct_metrics = _evaluate(direct_submission, ground_truth, args.map_num_workers)
    reference = direct_metrics["brief"]
    records = [{
        "config": direct_config,
        "selection_score": 1.0,
        "mean_selected_count": float(args.fixed_k),
        "brief": dict(reference),
    }]

    # Stage 3: learned duplicate probability, still under exactly the same K.
    for redundancy in args.redundancy_lambdas:
        config = {
            "mode": "learned_topk", "k": args.fixed_k,
            "redundancy_lambda": redundancy,
        }
        records.append(_record(
            config, decode(rows, config), ground_truth, args.map_num_workers, reference
        ))

    # Stage 4: count is a soft prior and selection may stop before max_output.
    soft_records = []
    for redundancy, count_weight, stop in itertools.product(
        args.redundancy_lambdas, args.count_prior_weights, args.stop_thresholds
    ):
        config = {
            "mode": "learned_soft_count", "max_output": args.max_output,
            "redundancy_lambda": redundancy,
            "count_prior_weight": count_weight,
            "stop_threshold": stop,
        }
        record = _record(
            config, decode(rows, config), ground_truth, args.map_num_workers, reference
        )
        records.append(record)
        soft_records.append(record)
    soft_records.sort(key=lambda item: item["selection_score"], reverse=True)
    best_soft = soft_records[0]["config"]

    # Stage 5: only the best soft selector receives the cautious fusion sweep.
    fusion_records = []
    for same_threshold, boundary_std in itertools.product(
        args.same_event_thresholds, args.boundary_std_thresholds
    ):
        config = {
            **best_soft,
            "mode": "learned_soft_count_fusion",
            "same_event_threshold": same_threshold,
            "boundary_std_threshold": boundary_std,
        }
        record = _record(
            config, decode(rows, config), ground_truth, args.map_num_workers, reference
        )
        records.append(record)
        fusion_records.append(record)

    learned_fixed = [row for row in records if row["config"]["mode"] == "learned_topk"]
    learned_fixed.sort(key=lambda item: item["selection_score"], reverse=True)
    fusion_records.sort(key=lambda item: item["selection_score"], reverse=True)
    best_by_stage = {
        "direct_topk": records[0],
        "learned_topk": learned_fixed[0],
        "learned_soft_count": soft_records[0],
        "learned_soft_count_fusion": fusion_records[0],
    }
    for stage, record in best_by_stage.items():
        submission = decode(rows, record["config"])
        prediction_output = output_dir / f"best_{stage}.predictions.jsonl"
        metrics_output = output_dir / f"best_{stage}.metrics.json"
        _write_jsonl(prediction_output, submission)
        _write_json(metrics_output, _evaluate(
            submission, ground_truth, args.map_num_workers
        ))
        record["prediction_output"] = str(prediction_output)
        record["metrics_output"] = str(metrics_output)

    summary = {
        "protocol": {
            "split": "validation", "test_labels_read": False,
            "fixed_k": args.fixed_k,
            "fair_fixed_k_comparison": ["direct_topk", "learned_topk"],
            "selection_rule": "min(mAP/direct_mAP, G-mIoU@3/direct_G-mIoU@3)",
        },
        "input_prediction": str(prediction_path),
        "ground_truth": str(gt_path),
        "best_by_stage": best_by_stage,
        "grid_size": len(records),
        "grid_records": sorted(
            records, key=lambda item: item["selection_score"], reverse=True
        ),
    }
    _write_json(output_dir / "learned_selector_ablation_summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ablate learned pairwise and soft-count selection on validation.",
        allow_abbrev=False,
    )
    parser.add_argument("--prediction-path", required=True)
    parser.add_argument("--gt-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fixed-k", type=int, default=3)
    parser.add_argument("--max-output", type=int, default=10)
    parser.add_argument("--redundancy-lambdas", type=float, nargs="+", default=[0.25, 0.5, 1.0, 2.0])
    parser.add_argument("--count-prior-weights", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0])
    parser.add_argument("--stop-thresholds", type=float, nargs="+", default=[-2.0, -1.5, -1.0, -0.5])
    parser.add_argument("--same-event-thresholds", type=float, nargs="+", default=[0.7, 0.8, 0.9])
    parser.add_argument("--boundary-std-thresholds", type=float, nargs="+", default=[0.02, 0.03, 0.05])
    parser.add_argument("--map-num-workers", type=int, default=1)
    return parser


def main() -> None:
    summary = run(build_parser().parse_args())
    compact = {
        stage: record["brief"]
        for stage, record in summary["best_by_stage"].items()
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
