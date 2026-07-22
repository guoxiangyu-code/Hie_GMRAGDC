#!/usr/bin/env python3
"""Audit null/single/multi behavior so G-mIoU gains cannot hide null collapse."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.metrics import _clean_pred_windows, _compute_set_iou_score, get_existence_score
from eval.normalization import normalize_ground_truth
from eval.utils import load_jsonl


def group_name(row: dict) -> str:
    count = len(row.get("relevant_windows", []))
    if count == 0:
        return "null"
    if count == 1:
        return "single"
    return "multi"


def diagnose(submission: list[dict], ground_truth: list[dict], *, threshold: float,
             k: int = 3) -> dict:
    predictions = {row["qid"]: row for row in submission}
    expected = {row["qid"] for row in ground_truth}
    if predictions.keys() != expected:
        raise ValueError(f"submission/GT qid coverage differs: {len(predictions)}/{len(expected)}")

    values = defaultdict(list)
    accepted = defaultdict(list)
    duplicate = defaultdict(list)
    for target in ground_truth:
        group = group_name(target)
        prediction = predictions[target["qid"]]
        existence, _ = get_existence_score(prediction)
        is_accepted = float(existence > threshold)
        raw = _clean_pred_windows(
            prediction.get("pred_relevant_windows", []), max_pred_windows=k
        )
        selected = raw if is_accepted else []
        pred_spans = [[window[0], window[1]] for window in selected]
        gt_spans = target.get("relevant_windows", [])
        set_iou = _compute_set_iou_score(pred_spans, gt_spans)
        raw_iou = _compute_set_iou_score(
            [[window[0], window[1]] for window in raw], gt_spans
        )
        values[group].append((set_iou, raw_iou))
        accepted[group].append(is_accepted)

        overlaps = []
        for first in range(len(raw)):
            a0, a1 = raw[first][:2]
            for second in range(first + 1, len(raw)):
                b0, b1 = raw[second][:2]
                intersection = max(0.0, min(a1, b1) - max(a0, b0))
                union = max(a1, b1) - min(a0, b0)
                overlaps.append(intersection / union if union > 0 else 0.0)
        duplicate[group].append(float(np.mean(np.asarray(overlaps) >= 0.5)) if overlaps else 0.0)

    report = {
        "threshold": threshold,
        "k": k,
        "num_queries": len(ground_truth),
        "groups": {},
    }
    for group in ("null", "single", "multi"):
        group_values = values[group]
        report["groups"][group] = {
            "support": len(group_values),
            "G-set-IoU@k": round(float(np.mean([row[0] for row in group_values]) * 100), 4),
            "raw-set-IoU@k": round(float(np.mean([row[1] for row in group_values]) * 100), 4),
            "acceptance_rate": round(float(np.mean(accepted[group]) * 100), 4),
            "topk_duplicate_pair_rate": round(float(np.mean(duplicate[group]) * 100), 4),
        }
    positive_scores = values["single"] + values["multi"]
    null_rejection = 1.0 - float(np.mean(accepted["null"]))
    positive_g = float(np.mean([row[0] for row in positive_scores]))
    report["balanced_G"] = round((null_rejection + positive_g) * 50.0, 4)
    report["all_empty_reference"] = {
        "G-mIoU": round(len(values["null"]) / len(ground_truth) * 100, 4),
        "balanced_G": 50.0,
    }
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--submission", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--k", type=int, default=3)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    ground_truth, _ = normalize_ground_truth(
        load_jsonl(args.ground_truth), None, drop_empty_gt=False
    )
    report = diagnose(
        load_jsonl(args.submission), ground_truth, threshold=args.threshold, k=args.k
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
