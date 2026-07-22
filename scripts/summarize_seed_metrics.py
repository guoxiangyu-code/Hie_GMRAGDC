#!/usr/bin/env python3
"""Summarize frozen multi-seed validation metrics against one matched anchor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def brief_metrics(path: str | Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    return value.get("brief", value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--keys", nargs="+", default=["mAP", "G-mIoU@3"])
    args = parser.parse_args()
    if len(args.metrics) != len(args.seeds):
        raise ValueError("--metrics and --seeds must have equal length")

    baseline = brief_metrics(args.baseline)
    rows = []
    for seed, path in zip(args.seeds, args.metrics):
        metrics = brief_metrics(path)
        rows.append({
            "seed": seed,
            "path": str(Path(path).resolve()),
            **{key: float(metrics[key]) for key in args.keys},
        })
    summary = {}
    for key in args.keys:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        reference = float(baseline[key])
        summary[key] = {
            "baseline": reference,
            "mean": round(float(values.mean()), 4),
            "sample_std": round(float(values.std(ddof=1)), 4) if len(values) > 1 else 0.0,
            "min": round(float(values.min()), 4),
            "max": round(float(values.max()), 4),
            "mean_delta": round(float(values.mean() - reference), 4),
            "all_seeds_improve": bool(np.all(values > reference)),
        }
    report = {"seeds": rows, "summary": summary}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
