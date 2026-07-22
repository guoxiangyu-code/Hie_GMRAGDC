#!/usr/bin/env python3
"""Compose localization proposals and GMR decision heads from frozen runs.

The GMR adapter is parallel to the DETR localization decoder.  This utility
therefore permits objective-specific checkpoint selection without mixing the
semantics of either output: windows and their scores come exclusively from the
localization submission, while existence/count fields come exclusively from
the decision submission.  Exact qid coverage and SHA256 provenance are
recorded so this cannot silently become an ad-hoc or test-tuned merge.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.eval_main import evaluate_gmr
from eval.normalization import normalize_ground_truth
from eval.utils import load_jsonl


DECISION_FIELDS = (
    "pred_exist_score",
    "pred_count",
    "pred_count_probs",
    "pred_exist_logit",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _index_unique(rows: list[dict], name: str) -> dict:
    index = {}
    for row in rows:
        if "qid" not in row:
            raise ValueError(f"{name} row is missing qid")
        qid = row["qid"]
        if qid in index:
            raise ValueError(f"{name} contains duplicate qid={qid!r}")
        index[qid] = row
    return index


def fuse_submissions(localization: list[dict], decision: list[dict]) -> list[dict]:
    localization_by_qid = _index_unique(localization, "localization")
    decision_by_qid = _index_unique(decision, "decision")
    if localization_by_qid.keys() != decision_by_qid.keys():
        missing_decision = sorted(localization_by_qid.keys() - decision_by_qid.keys())
        missing_localization = sorted(decision_by_qid.keys() - localization_by_qid.keys())
        raise ValueError(
            "qid coverage differs: "
            f"missing_decision={missing_decision[:5]}, "
            f"missing_localization={missing_localization[:5]}"
        )

    fused = []
    for localization_row in localization:
        qid = localization_row["qid"]
        decision_row = decision_by_qid[qid]
        if "pred_relevant_windows" not in localization_row:
            raise ValueError(f"localization qid={qid!r} has no pred_relevant_windows")
        if "pred_exist_score" not in decision_row:
            raise ValueError(f"decision qid={qid!r} has no pred_exist_score")
        row = dict(localization_row)
        for field in DECISION_FIELDS:
            if field in decision_row:
                row[field] = decision_row[field]
            else:
                row.pop(field, None)
        fused.append(row)
    return fused


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fuse frozen DETR localization and GMR decision submissions",
        allow_abbrev=False,
    )
    parser.add_argument("--localization", required=True)
    parser.add_argument("--decision", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--ground-truth", default=None)
    parser.add_argument("--metrics-output", default=None)
    parser.add_argument("--gmiou-threshold", type=float, default=0.4)
    parser.add_argument("--map-num-workers", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    localization_path = Path(args.localization)
    decision_path = Path(args.decision)
    output_path = Path(args.output)
    fused = fuse_submissions(
        load_jsonl(str(localization_path)), load_jsonl(str(decision_path))
    )
    write_jsonl(fused, output_path)

    manifest_path = Path(args.manifest) if args.manifest else output_path.with_suffix(
        output_path.suffix + ".manifest.json"
    )
    manifest = {
        "operation": "objective_specific_gmr_head_composition",
        "num_queries": len(fused),
        "localization": {
            "path": str(localization_path.resolve()),
            "sha256": sha256_file(localization_path),
            "owned_fields": ["pred_relevant_windows"],
        },
        "decision": {
            "path": str(decision_path.resolve()),
            "sha256": sha256_file(decision_path),
            "owned_fields": list(DECISION_FIELDS),
        },
        "output": {
            "path": str(output_path.resolve()),
            "sha256": sha256_file(output_path),
        },
        "gmiou_threshold": args.gmiou_threshold,
    }

    if args.ground_truth:
        ground_truth_path = Path(args.ground_truth)
        ground_truth, _ = normalize_ground_truth(
            load_jsonl(str(ground_truth_path)), None, drop_empty_gt=False
        )
        expected_qids = {row["qid"] for row in ground_truth}
        fused_qids = {row["qid"] for row in fused}
        if fused_qids != expected_qids:
            raise ValueError(
                f"fused/ground-truth qid coverage differs: "
                f"{len(fused_qids)}/{len(expected_qids)}"
            )
        metrics = evaluate_gmr(
            fused,
            ground_truth,
            gmiou_cls_threshold=args.gmiou_threshold,
            map_num_workers=args.map_num_workers,
            verbose=False,
        )
        metrics_path = (
            Path(args.metrics_output)
            if args.metrics_output
            else output_path.with_suffix(output_path.suffix + ".metrics.json")
        )
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, ensure_ascii=False, indent=2)
        manifest["ground_truth"] = {
            "path": str(ground_truth_path.resolve()),
            "sha256": sha256_file(ground_truth_path),
        }
        manifest["metrics"] = {
            "path": str(metrics_path.resolve()),
            "sha256": sha256_file(metrics_path),
            "brief": metrics["brief"],
        }
    elif args.metrics_output:
        raise ValueError("--metrics-output requires --ground-truth")

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
