#!/usr/bin/env python3
"""Freeze a validation-selected two-head GMR test protocol before test inference."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

import torch


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: str | Path) -> dict:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def brief(path: str | Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    return value.get("brief", value)


def checkpoint_metadata(path: str | Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    result = artifact(path)
    result["epoch_zero_based"] = checkpoint.get("epoch") if isinstance(checkpoint, dict) else None
    if isinstance(checkpoint, dict) and checkpoint.get("opt") is not None:
        option = checkpoint["opt"]
        result["seed"] = getattr(option, "seed", None)
        result["variant"] = getattr(option, "variant", None)
        result["round_to_clip"] = getattr(option, "round_to_clip", None)
        result["trim_text_by_attention_mask"] = getattr(
            option, "trim_text_by_attention_mask", None
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--localization-checkpoint", required=True)
    parser.add_argument("--decision-checkpoint", required=True)
    parser.add_argument("--fused-val-metrics", required=True)
    parser.add_argument("--baseline-val-metrics", required=True)
    parser.add_argument("--three-seed-summary", required=True)
    parser.add_argument("--val-annotations", required=True)
    parser.add_argument("--test-annotations", required=True)
    parser.add_argument("--source", nargs="+", required=True)
    parser.add_argument("--expected-test-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gmiou-threshold", type=float, default=0.4)
    parser.add_argument("--quality-alpha", type=float, default=0.5)
    parser.add_argument("--diversity-lambda", type=float, default=0.0)
    args = parser.parse_args()

    expected_test_dir = Path(args.expected_test_dir)
    existing_outputs = (
        [path for path in expected_test_dir.rglob("*") if path.is_file()]
        if expected_test_dir.exists() else []
    )
    if existing_outputs:
        raise RuntimeError(
            "test output directory is not pristine: "
            f"{[str(path) for path in existing_outputs[:5]]}"
        )

    candidate = brief(args.fused_val_metrics)
    baseline = brief(args.baseline_val_metrics)
    gates = {
        "mAP_improves": float(candidate["mAP"]) > float(baseline["mAP"]),
        "G-mIoU@3_improves": (
            float(candidate["G-mIoU@3"]) > float(baseline["G-mIoU@3"])
        ),
        "mR+@5_drop_le_0.5": (
            float(candidate.get("mR+@5", 0.0))
            >= float(baseline.get("mR+@5", 0.0)) - 0.5
        ),
    }
    three_seed = json.loads(Path(args.three_seed_summary).read_text(encoding="utf-8"))
    for key in ("mAP", "G-mIoU@3"):
        gates[f"three_seed_{key}_mean_improves"] = (
            float(three_seed["summary"][key]["mean_delta"]) > 0
        )
        gates[f"three_seed_{key}_all_improve"] = bool(
            three_seed["summary"][key]["all_seeds_improve"]
        )
    if not all(gates.values()):
        raise RuntimeError(f"validation gates failed: {gates}")

    source_manifest = []
    for value in args.source:
        path = Path(value)
        if path.is_dir():
            files = sorted(
                child for child in path.rglob("*")
                if child.is_file() and "__pycache__" not in child.parts
            )
            source_manifest.extend(artifact(child) for child in files)
        else:
            source_manifest.append(artifact(path))

    manifest = {
        "status": "frozen_before_test",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "selection_split": "validation",
        "primary_decode_mode": "full",
        "composition": {
            "localization_source": "mAP-best checkpoint",
            "decision_source": "joint-best checkpoint",
            "field_ownership": {
                "localization": ["pred_relevant_windows"],
                "decision": [
                    "pred_exist_score", "pred_count", "pred_count_probs",
                ],
            },
        },
        "fixed_inference": {
            "gmiou_threshold": args.gmiou_threshold,
            "quality_alpha": args.quality_alpha,
            "diversity_lambda": args.diversity_lambda,
            "round_to_clip": True,
            "max_predictions": 10,
        },
        "checkpoints": {
            "localization": checkpoint_metadata(args.localization_checkpoint),
            "decision": checkpoint_metadata(args.decision_checkpoint),
        },
        "validation": {
            "candidate_metrics": artifact(args.fused_val_metrics),
            "candidate_brief": candidate,
            "baseline_metrics": artifact(args.baseline_val_metrics),
            "baseline_brief": baseline,
            "three_seed_summary": artifact(args.three_seed_summary),
            "gates": gates,
            "annotations": artifact(args.val_annotations),
        },
        # The test file is hashed bytewise only; no metric or label-dependent
        # choice is made while creating this manifest.
        "test": {
            "annotations": artifact(args.test_annotations),
            "expected_output_dir": str(expected_test_dir.resolve()),
            "evaluation_count_before_freeze": 0,
        },
        "source_files": source_manifest,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
