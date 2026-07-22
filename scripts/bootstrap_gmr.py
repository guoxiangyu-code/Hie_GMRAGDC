#!/usr/bin/env python3
"""Paired validation bootstrap for mAP and G-mIoU@3 deltas."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.metrics import (
    DEFAULT_IOU_THRESHOLDS,
    _ap_worker,
    _clean_pred_windows,
    _compute_set_iou_score,
    get_existence_score,
)
from eval.normalization import normalize_ground_truth
from eval.utils import load_jsonl


def per_query_ap(submission, positive_gt):
    pred_by_qid = {row["qid"]: row for row in submission}
    values = []
    for row in positive_gt:
        qid = row["qid"]
        gt_items = [
            {"video-id": qid, "t-start": window[0], "t-end": window[1]}
            for window in row["relevant_windows"]
        ]
        pred_items = [
            {
                "video-id": qid,
                "t-start": window[0],
                "t-end": window[1],
                "score": window[2],
            }
            for window in _clean_pred_windows(
                pred_by_qid.get(qid, {}).get("pred_relevant_windows", []),
                max_pred_windows=10,
            )
        ]
        _, scores = _ap_worker((qid, gt_items, pred_items), DEFAULT_IOU_THRESHOLDS)
        values.append(float(np.mean(scores)))
    return np.asarray(values, dtype=np.float64)


def per_query_gmiou3(submission, ground_truth, threshold):
    pred_by_qid = {row["qid"]: row for row in submission}
    values = []
    for row in ground_truth:
        prediction = pred_by_qid.get(row["qid"], {})
        score, _ = get_existence_score(prediction)
        windows = _clean_pred_windows(
            prediction.get("pred_relevant_windows", []), max_pred_windows=3
        ) if score > threshold else []
        values.append(_compute_set_iou_score(
            [[window[0], window[1]] for window in windows],
            row["relevant_windows"],
        ))
    return np.asarray(values, dtype=np.float64)


def summarize_delta(delta, rng, samples):
    observed = float(delta.mean())
    draws = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 1000):
        size = min(1000, samples - start)
        indices = rng.integers(0, len(delta), size=(size, len(delta)))
        draws[start:start + size] = delta[indices].mean(axis=1)
    return {
        "delta": round(observed * 100, 4),
        "ci95": [
            round(float(np.quantile(draws, 0.025) * 100), 4),
            round(float(np.quantile(draws, 0.975) * 100), 4),
        ],
        "probability_delta_gt_zero": round(float(np.mean(draws > 0)), 6),
    }, draws


def main():
    parser = argparse.ArgumentParser(description="Paired bootstrap on a frozen validation split")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--ground_truth", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gmiou_threshold", type=float, default=0.4)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20230722)
    args = parser.parse_args()

    if "test" in Path(args.ground_truth).name.lower():
        raise ValueError("Refusing validation bootstrap on a test-labelled GT file")
    baseline = load_jsonl(args.baseline)
    candidate = load_jsonl(args.candidate)
    ground_truth, _ = normalize_ground_truth(
        load_jsonl(args.ground_truth), None, drop_empty_gt=False
    )
    expected = {row["qid"] for row in ground_truth}
    for name, submission in (("baseline", baseline), ("candidate", candidate)):
        actual = {row.get("qid") for row in submission}
        if actual != expected:
            raise ValueError(f"{name} qid coverage differs from GT: {len(actual)}/{len(expected)}")

    positive_gt = [row for row in ground_truth if row["relevant_windows"]]
    base_ap = per_query_ap(baseline, positive_gt)
    cand_ap = per_query_ap(candidate, positive_gt)
    base_g = per_query_gmiou3(baseline, ground_truth, args.gmiou_threshold)
    cand_g = per_query_gmiou3(candidate, ground_truth, args.gmiou_threshold)

    rng = np.random.default_rng(args.seed)
    map_summary, map_draws = summarize_delta(cand_ap - base_ap, rng, args.samples)
    g_summary, g_draws = summarize_delta(cand_g - base_g, rng, args.samples)
    report = {
        "split": "validation",
        "samples": args.samples,
        "seed": args.seed,
        "gmiou_threshold": args.gmiou_threshold,
        "num_positive": len(positive_gt),
        "num_all": len(ground_truth),
        "baseline": {
            "mAP": round(float(base_ap.mean() * 100), 4),
            "G-mIoU@3": round(float(base_g.mean() * 100), 4),
        },
        "candidate": {
            "mAP": round(float(cand_ap.mean() * 100), 4),
            "G-mIoU@3": round(float(cand_g.mean() * 100), 4),
        },
        "paired_delta": {"mAP": map_summary, "G-mIoU@3": g_summary},
        "probability_both_deltas_gt_zero": round(
            float(np.mean((map_draws > 0) & (g_draws > 0))), 6
        ),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
