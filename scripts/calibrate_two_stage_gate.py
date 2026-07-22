#!/usr/bin/env python3
"""Calibrate high-recall gate plus independent-null decisions on validation."""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.eval_main import evaluate_gmr
from eval.normalization import normalize_ground_truth
from eval.utils import load_jsonl
from models.moment_detr_gmr.learned_selector import two_stage_accept
from scripts.ablate_temporal_dedup import assert_validation_input_path


def run(args: argparse.Namespace) -> dict:
    prediction_path = assert_validation_input_path(args.prediction_path, role="prediction")
    gt_path = assert_validation_input_path(args.gt_path, role="ground truth")
    rows = load_jsonl(str(prediction_path))
    gt_raw = load_jsonl(str(gt_path))
    ground_truth, _ = normalize_ground_truth(gt_raw, None, drop_empty_gt=False)
    gt_by_qid = {row["qid"]: row for row in ground_truth}
    if {row["qid"] for row in rows} != set(gt_by_qid):
        raise ValueError("Prediction/validation ground-truth qid coverage differs")

    gate_scores = torch.tensor([float(row["pred_gate_score"]) for row in rows])
    zero_scores = torch.tensor([float(row["pred_zero_score"]) for row in rows])
    local_scores = torch.tensor([
        float(row["pred_localization_evidence"]) for row in rows
    ])
    positive = torch.tensor([
        bool(gt_by_qid[row["qid"]].get("relevant_windows")) for row in rows
    ])
    records = []
    for gate, zero, veto_gap, localization in itertools.product(
        args.gate_thresholds, args.zero_thresholds,
        args.veto_gaps, args.localization_thresholds,
    ):
        veto = min(0.99, zero + veto_gap)
        accepted = two_stage_accept(
            gate_scores, zero_scores, local_scores,
            gate_threshold=gate, zero_threshold=zero,
            veto_threshold=veto, localization_threshold=localization,
        )
        submission = copy.deepcopy(rows)
        for row, decision in zip(submission, accepted.tolist()):
            row["pred_exist_score"] = float(decision)
            row["pred_exist_decision"] = int(decision)
        metrics = evaluate_gmr(
            submission, ground_truth, k_list=(1, 3, 5), max_pred_windows=10,
            cls_thresholds=(0.4,), gmiou_cls_threshold=0.4,
            map_num_workers=args.map_num_workers, verbose=False,
        )
        positive_pass = float(accepted[positive].float().mean()) if positive.any() else 0.0
        negative_reject = float((~accepted[~positive]).float().mean()) if (~positive).any() else 0.0
        records.append({
            "config": {
                "gate_recall_thd": gate, "zero_decision_thd": zero,
                "zero_veto_thd": veto,
                "zero_localization_thd": localization,
            },
            "positive_pass_rate": positive_pass,
            "negative_rejection_rate": negative_reject,
            "brief": dict(metrics["brief"]),
        })

    eligible = [
        row for row in records
        if row["positive_pass_rate"] >= args.minimum_positive_pass_rate
    ]
    pool = eligible or records
    pool.sort(key=lambda row: (
        float(row["brief"]["G-mIoU@3"]),
        float(row["brief"]["Rej-F1@0.4"]),
        row["negative_rejection_rate"],
    ), reverse=True)
    pareto = []
    for candidate in records:
        dominated = any(
            other is not candidate
            and other["positive_pass_rate"] >= candidate["positive_pass_rate"]
            and other["negative_rejection_rate"] >= candidate["negative_rejection_rate"]
            and float(other["brief"]["G-mIoU@3"]) >= float(candidate["brief"]["G-mIoU@3"])
            and (
                other["positive_pass_rate"] > candidate["positive_pass_rate"]
                or other["negative_rejection_rate"] > candidate["negative_rejection_rate"]
                or float(other["brief"]["G-mIoU@3"]) > float(candidate["brief"]["G-mIoU@3"])
            )
            for other in records
        )
        if not dominated:
            pareto.append(candidate)
    manifest = {
        "protocol": {
            "split": "validation", "test_labels_read": False,
            "minimum_positive_pass_rate": args.minimum_positive_pass_rate,
            "selection": "max G-mIoU@3 subject to positive-pass constraint",
            "thresholded_AUROC_is_not_used_for_selection": True,
        },
        "selected": pool[0],
        "pareto": sorted(
            pareto, key=lambda row: float(row["brief"]["G-mIoU@3"]), reverse=True
        ),
        "grid_size": len(records),
        "grid_records": records,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--prediction-path", required=True)
    parser.add_argument("--gt-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gate-thresholds", type=float, nargs="+", default=[0.15, 0.2, 0.25, 0.3, 0.35])
    parser.add_argument("--zero-thresholds", type=float, nargs="+", default=[0.5, 0.6, 0.7, 0.8])
    parser.add_argument("--veto-gaps", type=float, nargs="+", default=[0.05, 0.1, 0.15])
    parser.add_argument("--localization-thresholds", type=float, nargs="+", default=[0.1, 0.15, 0.2, 0.25])
    parser.add_argument("--minimum-positive-pass-rate", type=float, default=0.95)
    parser.add_argument("--map-num-workers", type=int, default=1)
    return parser


def main() -> None:
    result = run(build_parser().parse_args())
    print(json.dumps(result["selected"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
