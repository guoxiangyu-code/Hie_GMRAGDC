#!/usr/bin/env python3
"""Validation-only calibration for HieA2M adaptive set decoding.

The script consumes a ``--save_raw_queries`` validation submission, performs a
predeclared coarse grid over ranking/cardinality controls, and writes a frozen
configuration plus its exact input hashes.  It must never be run on test GT.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import re
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.eval_main import evaluate_gmr
from eval.normalization import normalize_ground_truth
from eval.utils import load_jsonl
from models.moment_detr_gmr.set_decoder import adaptive_count_indices, diversity_ranking


CALIBRATION_CONTRACT_VERSION = 2
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(path: str, *, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"{label} must be an existing regular file: {resolved}")
    return resolved


def _load_json(path: str, *, label: str):
    resolved = _regular_file(path, label=label)
    try:
        with resolved.open("r", encoding="utf-8") as handle:
            return resolved, json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} must be valid UTF-8 JSON: {resolved}") from error


def build_annotation_provenance(
    ground_truth_path: str,
    *,
    identity: str,
    role: str,
    expected_sha256: str,
) -> dict:
    """Validate and bind the exact validation annotation bytes.

    A filename is not evidence of split identity.  The caller must explicitly
    declare the role and stable identity and pin the expected file digest.
    """

    if role != "validation":
        raise ValueError(
            f"Calibration annotation_role must be 'validation', got {role!r}"
        )
    identity = str(identity).strip()
    if not identity:
        raise ValueError("annotation_identity must be a non-empty stable identifier")
    expected_sha256 = str(expected_sha256).strip().lower()
    if not _SHA256_PATTERN.fullmatch(expected_sha256):
        raise ValueError("annotation_sha256 must be exactly 64 lowercase hex characters")
    resolved = _regular_file(ground_truth_path, label="ground truth annotation")
    actual_sha256 = sha256(str(resolved))
    if actual_sha256 != expected_sha256:
        raise ValueError(
            "Ground-truth annotation digest does not match the declared identity: "
            f"expected={expected_sha256} actual={actual_sha256}"
        )
    return {
        "identity": identity,
        "role": role,
        "path": str(resolved),
        "sha256": actual_sha256,
    }


def build_producer_provenance(
    checkpoint_path: str,
    argv_json_path: str,
    source_files: list[str],
) -> dict:
    """Validate required raw-submission producer provenance and hash it."""

    checkpoint = _regular_file(checkpoint_path, label="producer checkpoint")
    argv_path, argv = _load_json(argv_json_path, label="producer argv artifact")
    if (
        not isinstance(argv, list)
        or not argv
        or any(not isinstance(value, str) or not value for value in argv)
    ):
        raise ValueError(
            "producer argv artifact must be a non-empty JSON array of non-empty strings"
        )
    if not source_files:
        raise ValueError("At least one producer_source_file is required")

    sources = []
    seen: set[Path] = set()
    for source_path in source_files:
        source = _regular_file(source_path, label="producer source file")
        if source in seen:
            raise ValueError(f"Duplicate producer source file: {source}")
        seen.add(source)
        sources.append({"path": str(source), "sha256": sha256(str(source))})

    return {
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": sha256(str(checkpoint)),
        },
        "argv": {
            "artifact_path": str(argv_path),
            "artifact_sha256": sha256(str(argv_path)),
            "value": argv,
        },
        "source_files": sources,
    }


def load_reference_metrics(path: str) -> dict:
    """Load the matched validation reference and bind its source artifact."""

    resolved, payload = _load_json(path, label="reference metrics artifact")
    if not isinstance(payload, dict):
        raise ValueError("reference metrics artifact must contain a JSON object")
    brief = payload.get("brief", payload)
    if not isinstance(brief, dict):
        raise ValueError("reference metrics artifact field 'brief' must be an object")
    try:
        reference_map = float(brief["mAP"])
        reference_gmiou3 = float(brief["G-mIoU@3"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "reference metrics artifact must provide numeric brief.mAP and "
            "brief.G-mIoU@3"
        ) from error
    if (
        not math.isfinite(reference_map)
        or not math.isfinite(reference_gmiou3)
        or reference_map <= 0
        or reference_gmiou3 <= 0
    ):
        raise ValueError("Matched validation reference metrics must be finite and positive")
    return {
        "artifact_path": str(resolved),
        "artifact_sha256": sha256(str(resolved)),
        "values": {"mAP": reference_map, "G-mIoU@3": reference_gmiou3},
    }


def _bound_file_hashes(annotation: dict, submission: dict, producer: dict, reference: dict):
    bound = {
        annotation["path"]: annotation["sha256"],
        submission["path"]: submission["sha256"],
        producer["checkpoint"]["path"]: producer["checkpoint"]["sha256"],
        producer["argv"]["artifact_path"]: producer["argv"]["artifact_sha256"],
        reference["artifact_path"]: reference["artifact_sha256"],
    }
    for source in producer["source_files"]:
        bound[source["path"]] = source["sha256"]
    return bound


def _verify_bound_files_unchanged(bound_hashes: dict[str, str]) -> None:
    for path, expected in bound_hashes.items():
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(
                f"Bound calibration input changed during execution: {path}; "
                f"before={expected} after={actual}"
            )


def save_json(data, path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def save_jsonl(data, path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(json.dumps(row, ensure_ascii=False) for row in data))


def _unique_qid_index(rows, source: str) -> dict[object, int]:
    index: dict[object, int] = {}
    for row_index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"{source} row {row_index} must be an object")
        if "qid" not in row or row["qid"] is None:
            raise ValueError(f"{source} row {row_index} is missing a non-null qid")
        qid = row["qid"]
        try:
            hash(qid)
        except TypeError as error:
            raise ValueError(
                f"{source} row {row_index} has an unhashable qid: {qid!r}"
            ) from error
        if qid in index:
            raise ValueError(
                f"{source} contains duplicate qid={qid!r} at rows "
                f"{index[qid]} and {row_index}"
            )
        index[qid] = row_index
    return index


def validate_qid_coverage(rows, ground_truth) -> None:
    pred_index = _unique_qid_index(rows, "submission")
    gt_index = _unique_qid_index(ground_truth, "ground truth")
    pred_qids = set(pred_index)
    gt_qids = set(gt_index)
    if pred_qids != gt_qids:
        missing = sorted(gt_qids - pred_qids, key=repr)
        extra = sorted(pred_qids - gt_qids, key=repr)
        raise ValueError(
            f"Incomplete validation coverage: pred={len(pred_qids)} gt={len(gt_qids)} "
            f"missing={missing} extra={extra}"
        )


def _float_tensor(value, *, field: str, qid) -> torch.Tensor:
    try:
        tensor = torch.as_tensor(value, dtype=torch.float32)
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError(f"qid={qid!r} field {field} is not a rectangular numeric array") from error
    if not torch.isfinite(tensor).all():
        raise ValueError(f"qid={qid!r} field {field} contains non-finite values")
    return tensor


def _matrix(value, *, field: str, qid, columns: int | tuple[int, ...]) -> torch.Tensor:
    tensor = _float_tensor(value, field=field, qid=qid)
    allowed = (columns,) if isinstance(columns, int) else tuple(columns)
    if tensor.ndim != 2 or tensor.shape[0] == 0 or tensor.shape[1] not in allowed:
        expected = " or ".join(f"N×{column}" for column in allowed)
        raise ValueError(
            f"qid={qid!r} field {field} must have shape {expected} with N>0; "
            f"got {tuple(tensor.shape)}"
        )
    return tensor


def _count_probabilities(value, *, qid) -> torch.Tensor:
    probabilities = _float_tensor(value, field="pred_count_probs", qid=qid)
    if probabilities.shape != (5,):
        raise ValueError(
            f"qid={qid!r} field pred_count_probs must have shape (5,); "
            f"got {tuple(probabilities.shape)}"
        )
    if (probabilities < 0).any() or (probabilities > 1).any():
        raise ValueError(f"qid={qid!r} field pred_count_probs must lie in [0,1]")
    if not torch.isclose(
        probabilities.sum(), probabilities.new_tensor(1.0), atol=1e-4, rtol=0.0
    ):
        raise ValueError(
            f"qid={qid!r} field pred_count_probs must sum to 1; "
            f"got {float(probabilities.sum()):.6f}"
        )
    return probabilities


def _raw_query_tensors(row, *, max_ts: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    qid = row.get("qid")
    components_raw = row.get("all_query_components")
    if components_raw is None:
        raise ValueError(
            f"qid={qid!r} is missing all_query_components; rerun evaluate.py "
            "with --save_raw_queries."
        )
    components = _matrix(
        components_raw, field="all_query_components", qid=qid, columns=(2, 4)
    )
    windows_raw = row.get("all_query_windows")
    windows = None
    if windows_raw is not None:
        windows = _matrix(
            windows_raw, field="all_query_windows", qid=qid, columns=3
        )
        if windows.shape[0] != components.shape[0]:
            raise ValueError(
                f"qid={qid!r} raw-query row mismatch: "
                f"all_query_windows={windows.shape[0]} "
                f"all_query_components={components.shape[0]}"
            )

    if components.shape[1] == 4:
        # Moment-DETR stores [start,end,foreground,quality] together.  Its
        # optional all_query_windows is ranking-ordered, while this matrix is
        # query-index ordered, so the two fields must not be zipped.
        spans = components[:, :2]
        foreground = components[:, 2]
        quality = components[:, 3]
    else:
        # EaTR/QD/CG store ranking-aligned [start,end,old_score] separately
        # from [foreground,quality].  The old score is deliberately ignored:
        # every calibration candidate recomputes it at its requested alpha.
        if windows is None:
            raise ValueError(
                f"qid={qid!r} 2-column all_query_components requires "
                "all_query_windows with shape N×3"
            )
        spans = windows[:, :2]
        foreground = components[:, 0]
        quality = components[:, 1]

    if (spans[:, 0] > spans[:, 1]).any():
        raise ValueError(f"qid={qid!r} raw-query span has start greater than end")
    if (spans < 0).any() or (spans > float(max_ts)).any():
        raise ValueError(
            f"qid={qid!r} raw-query spans must lie in [0,{float(max_ts)}]"
        )
    if (foreground < 0).any() or (foreground > 1).any():
        raise ValueError(f"qid={qid!r} foreground components must lie in [0,1]")
    if (quality < 0).any() or (quality > 1).any():
        raise ValueError(f"qid={qid!r} quality components must lie in [0,1]")
    return spans, foreground, quality


def decode_submission(rows, config, round_to_clip: bool, clip_length: float, max_ts: float):
    _unique_qid_index(rows, "submission")
    decoded = []
    for row in rows:
        qid = row["qid"]
        probabilities_raw = row.get("pred_count_probs")
        if probabilities_raw is None:
            raise ValueError(
                f"qid={qid!r} is missing pred_count_probs; calibration requires "
                "a hierarchical-counter submission."
            )
        spans, foreground, quality = _raw_query_tensors(row, max_ts=max_ts)
        count_values = _count_probabilities(probabilities_raw, qid=qid)
        foreground = foreground.clamp_min(torch.finfo(torch.float32).eps)
        quality = quality.clamp_min(torch.finfo(torch.float32).eps)
        alpha = float(config["quality_alpha"])
        scores = foreground.pow(1.0 - alpha) * quality.pow(alpha)
        ranking = diversity_ranking(
            spans,
            scores,
            diversity_lambda=float(config["diversity_lambda"]),
        )
        indices = adaptive_count_indices(
            ranking,
            scores,
            count_values,
            mode=str(config["decode_mode"]),
            existence_threshold=float(config["existence_threshold"]),
            count_confidence_threshold=float(config["count_confidence_threshold"]),
            window_score_threshold=float(config["window_score_threshold"]),
        )
        selected_spans = spans[indices]
        if round_to_clip and len(indices):
            selected_spans = selected_spans.clamp(0, max_ts)
            selected_spans = torch.round(selected_spans / clip_length) * clip_length
            selected_spans = selected_spans.clamp(0, max_ts)
        windows = [
            [
                float(f"{float(selected_spans[offset, 0]):.4f}"),
                float(f"{float(selected_spans[offset, 1]):.4f}"),
                float(f"{float(scores[index]):.4f}"),
            ]
            for offset, index in enumerate(indices)
        ]
        predicted_count = (
            int(torch.argmax(count_values[1:]).item()) + 1
            if 1.0 - float(count_values[0]) > float(config["existence_threshold"])
            else 0
        )
        decoded_row = {
            "qid": qid,
            "query": row.get("query", ""),
            "vid": row.get("vid", ""),
            "pred_relevant_windows": windows,
            "pred_exist_score": row.get("pred_exist_score", 1.0),
            "pred_count": predicted_count,
            "pred_count_probs": [float(value) for value in count_values],
        }
        decoded.append(decoded_row)
    return decoded


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calibrate HieA2M on a complete validation split.")
    parser.add_argument("--submission", required=True)
    parser.add_argument("--ground_truth", required=True)
    parser.add_argument("--annotation_identity", required=True)
    parser.add_argument("--annotation_role", required=True, choices=["validation"])
    parser.add_argument(
        "--annotation_sha256",
        required=True,
        help="Expected SHA-256 of the exact validation annotation file",
    )
    parser.add_argument("--producer_checkpoint", required=True)
    parser.add_argument(
        "--producer_argv_json",
        required=True,
        help="UTF-8 JSON file containing the producer argv as a string array",
    )
    parser.add_argument(
        "--producer_source_files",
        required=True,
        nargs="+",
        help="Exact source files used to produce the raw-query submission",
    )
    parser.add_argument(
        "--reference_metrics",
        required=True,
        help="Matched validation metrics JSON containing brief.mAP and brief.G-mIoU@3",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["full", "threshold", "adaptive"],
        default=["full", "threshold", "adaptive"],
    )
    parser.add_argument("--existence_thresholds", type=float, nargs="+", default=[0.4])
    parser.add_argument("--count_confidence_thresholds", type=float, nargs="+", default=[0.4, 0.55, 0.7])
    parser.add_argument("--window_score_thresholds", type=float, nargs="+", default=[0.05, 0.1, 0.2])
    parser.add_argument("--quality_alphas", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    parser.add_argument("--diversity_lambdas", type=float, nargs="+", default=[0.0, 0.25, 0.5])
    parser.add_argument("--round_to_clip", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--clip_length", type=float, default=2.0)
    parser.add_argument("--max_ts", type=float, default=150.0)
    parser.add_argument("--map_num_workers", type=int, default=1)
    return parser


def calibrate(args: argparse.Namespace) -> dict:
    annotation = build_annotation_provenance(
        args.ground_truth,
        identity=args.annotation_identity,
        role=args.annotation_role,
        expected_sha256=args.annotation_sha256,
    )
    submission_path = _regular_file(args.submission, label="raw-query submission")
    submission_provenance = {
        "path": str(submission_path),
        "sha256": sha256(str(submission_path)),
    }
    producer = build_producer_provenance(
        args.producer_checkpoint,
        args.producer_argv_json,
        args.producer_source_files,
    )
    reference = load_reference_metrics(args.reference_metrics)
    reference_map = float(reference["values"]["mAP"])
    reference_gmiou3 = float(reference["values"]["G-mIoU@3"])
    bound_hashes = _bound_file_hashes(
        annotation, submission_provenance, producer, reference
    )

    if len(set(args.existence_thresholds)) != 1:
        raise ValueError(
            "Calibrate one existence threshold per run so reference_gmiou3 "
            "comes from the same threshold protocol"
        )

    rows = load_jsonl(str(submission_path))
    gt_raw = load_jsonl(annotation["path"])
    _unique_qid_index(gt_raw, "ground truth")
    ground_truth, _ = normalize_ground_truth(gt_raw, None, drop_empty_gt=False)
    validate_qid_coverage(rows, ground_truth)

    grid = []
    if "full" in args.modes:
        grid.extend(
            ("full", existence, 0.0, 0.0, alpha, diversity)
            for existence, alpha, diversity in itertools.product(
                args.existence_thresholds,
                args.quality_alphas,
                args.diversity_lambdas,
            )
        )
    if "adaptive" in args.modes:
        grid.extend(
            ("adaptive", existence, count_confidence, window_score, alpha, diversity)
            for existence, count_confidence, window_score, alpha, diversity in itertools.product(
                args.existence_thresholds,
                args.count_confidence_thresholds,
                args.window_score_thresholds,
                args.quality_alphas,
                args.diversity_lambdas,
            )
        )
    if "threshold" in args.modes:
        grid.extend(
            ("threshold", existence, 0.0, window_score, alpha, diversity)
            for existence, window_score, alpha, diversity in itertools.product(
                args.existence_thresholds,
                args.window_score_thresholds,
                args.quality_alphas,
                args.diversity_lambdas,
            )
        )
    records = []
    for mode, existence, count_confidence, window_score, alpha, diversity in grid:
        config = {
            "decode_mode": mode,
            "existence_threshold": existence,
            "count_confidence_threshold": count_confidence,
            "window_score_threshold": window_score,
            "quality_alpha": alpha,
            "diversity_lambda": diversity,
        }
        submission = decode_submission(
            rows,
            config,
            round_to_clip=args.round_to_clip,
            clip_length=args.clip_length,
            max_ts=args.max_ts,
        )
        metrics = evaluate_gmr(
            submission,
            ground_truth,
            k_list=(1, 3, 5),
            cls_thresholds=(0.4, 0.6, 0.8),
            gmiou_cls_threshold=float(existence),
            map_num_workers=args.map_num_workers,
            verbose=False,
        )
        brief = metrics["brief"]
        score = min(
            float(brief["mAP"]) / reference_map,
            float(brief["G-mIoU@3"]) / reference_gmiou3,
        )
        records.append({
            "config": config,
            "score": score,
            "simultaneous_gain": (
                float(brief["mAP"]) > reference_map
                and float(brief["G-mIoU@3"]) > reference_gmiou3
            ),
            "brief": brief,
        })

    records.sort(key=lambda row: row["score"], reverse=True)
    best_config = records[0]
    best_by_mode = {
        mode: next(row for row in records if row["config"]["decode_mode"] == mode)
        for mode in args.modes
    }
    # Recreate after sorting so the saved prediction unambiguously matches the
    # selected configuration rather than the pre-sort record index.
    selected_submission = decode_submission(
        rows,
        best_config["config"],
        round_to_clip=args.round_to_clip,
        clip_length=args.clip_length,
        max_ts=args.max_ts,
    )
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    output_path = Path(output)
    selected_path = output_path.with_name(f"{output_path.stem}_best_submission.jsonl")
    save_jsonl(selected_submission, str(selected_path))
    selected_provenance = {
        "path": str(selected_path.resolve()),
        "sha256": sha256(str(selected_path)),
        "row_count": len(selected_submission),
    }
    _verify_bound_files_unchanged(bound_hashes)
    manifest = {
        "contract_version": CALIBRATION_CONTRACT_VERSION,
        "calibration_split": {
            "identity": annotation["identity"],
            "role": annotation["role"],
        },
        "ground_truth": annotation,
        "input_submission": submission_provenance,
        "producer": producer,
        "reference_metrics": reference,
        "selected_submission": selected_provenance,
        "calibrator": {
            "source_path": str(Path(__file__).resolve()),
            "source_sha256": sha256(str(Path(__file__).resolve())),
        },
        "round_to_clip": args.round_to_clip,
        "clip_length": args.clip_length,
        "max_ts": args.max_ts,
        "grid_size": len(records),
        "selected": best_config,
        "best_by_mode": best_by_mode,
        "grid_records": records,
        "top10": records[:10],
    }
    save_json(manifest, output)
    return manifest


def main() -> None:
    args = build_parser().parse_args()
    best_config = calibrate(args)["selected"]
    print(json.dumps(best_config, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
