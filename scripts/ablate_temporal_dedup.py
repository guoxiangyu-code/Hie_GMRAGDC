#!/usr/bin/env python3
"""Validation-only temporal deduplication ablations for Soccer-GMR predictions.

The tool deliberately operates after inference.  It preserves the model's
existence/count outputs, changes only ``pred_relevant_windows``, and evaluates
each generated submission with the repository's canonical ``evaluate_gmr``
API.  Test-labelled inputs are rejected before any file is opened.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.eval_main import evaluate_gmr
from eval.metrics import DEFAULT_IOU_THRESHOLDS
from eval.normalization import load_ts_window_cfg, normalize_ground_truth
from eval.utils import load_jsonl


METHODS = (
    "none",
    "hard_nms",
    "diou_nms",
    "soft_nms_linear",
    "soft_nms_gaussian",
    "cluster_representative",
    "cluster_fusion",
)


@dataclass(frozen=True)
class DedupSpec:
    method: str
    iou_threshold: float | None = None
    sigma: float | None = None
    score_floor: float | None = None
    cluster_linkage: str | None = None


def temporal_iou(first: Sequence[float], second: Sequence[float]) -> float:
    """Return IoU for two valid ``[start, end, ...]`` temporal windows."""
    intersection = max(0.0, min(float(first[1]), float(second[1])) - max(float(first[0]), float(second[0])))
    union = (
        float(first[1]) - float(first[0])
        + float(second[1]) - float(second[0])
        - intersection
    )
    return intersection / union if union > 0 else 0.0


def _score_sorted(windows: Sequence[Sequence[float]]) -> list[list[float]]:
    indexed = [(index, list(window)) for index, window in enumerate(windows)]
    indexed.sort(key=lambda item: (-float(item[1][2]), item[0]))
    return [window for _, window in indexed]


def hard_temporal_nms(
    windows: Sequence[Sequence[float]],
    iou_threshold: float,
) -> list[list[float]]:
    """Greedy hard temporal NMS, matching the repository's ``IoU > threshold`` rule."""
    remaining = _score_sorted(windows)
    selected: list[list[float]] = []
    while remaining:
        best = remaining.pop(0)
        selected.append(best)
        remaining = [
            candidate
            for candidate in remaining
            if temporal_iou(best, candidate) <= iou_threshold
        ]
    return selected


def temporal_diou(first: Sequence[float], second: Sequence[float]) -> float:
    """One-dimensional DIoU similarity used by the DIoU-NMS ablation."""
    iou = temporal_iou(first, second)
    first_center = 0.5 * (float(first[0]) + float(first[1]))
    second_center = 0.5 * (float(second[0]) + float(second[1]))
    enclosing = max(float(first[1]), float(second[1])) - min(
        float(first[0]), float(second[0])
    )
    return iou - ((first_center - second_center) / max(enclosing, 1e-8)) ** 2


def diou_temporal_nms(
    windows: Sequence[Sequence[float]],
    threshold: float,
) -> list[list[float]]:
    remaining = _score_sorted(windows)
    selected: list[list[float]] = []
    while remaining:
        best = remaining.pop(0)
        selected.append(best)
        remaining = [
            candidate for candidate in remaining
            if temporal_diou(best, candidate) <= threshold
        ]
    return selected


def soft_temporal_nms(
    windows: Sequence[Sequence[float]],
    *,
    mode: str,
    iou_threshold: float = 0.5,
    sigma: float = 0.5,
    score_floor: float = 1e-4,
) -> list[list[float]]:
    """Iterative linear or Gaussian Soft-NMS with score re-sorting."""
    if mode not in {"linear", "gaussian"}:
        raise ValueError(f"Unsupported Soft-NMS mode: {mode}")
    if sigma <= 0:
        raise ValueError("Soft-NMS sigma must be positive")
    if score_floor < 0:
        raise ValueError("Soft-NMS score_floor must be non-negative")

    working = [
        {"window": list(window), "order": index}
        for index, window in enumerate(windows)
    ]
    selected: list[list[float]] = []
    while working:
        working.sort(key=lambda item: (-float(item["window"][2]), int(item["order"])))
        best_item = working.pop(0)
        best = list(best_item["window"])
        selected.append(best)

        updated = []
        for item in working:
            candidate = list(item["window"])
            overlap = temporal_iou(best, candidate)
            if mode == "linear":
                decay = 1.0 - overlap if overlap > iou_threshold else 1.0
            else:
                decay = math.exp(-(overlap * overlap) / sigma)
            candidate[2] = float(candidate[2]) * decay
            if candidate[2] >= score_floor:
                updated.append({"window": candidate, "order": item["order"]})
        working = updated
    return selected


def complete_link_clusters(
    windows: Sequence[Sequence[float]],
    iou_threshold: float,
) -> list[list[list[float]]]:
    """Greedily form complete-link clusters in descending score order.

    A candidate joins a cluster only if it overlaps *every* member above the
    threshold.  This avoids the transitive chaining of connected-components or
    single-link clustering, which can merge two distinct nearby events through
    an intermediate proposal.
    """
    clusters: list[list[list[float]]] = []
    for candidate in _score_sorted(windows):
        compatible: list[tuple[float, int]] = []
        for cluster_index, cluster in enumerate(clusters):
            overlaps = [temporal_iou(candidate, member) for member in cluster]
            if overlaps and min(overlaps) > iou_threshold:
                compatible.append((sum(overlaps) / len(overlaps), cluster_index))
        if compatible:
            _, best_cluster = max(compatible, key=lambda item: (item[0], -item[1]))
            clusters[best_cluster].append(candidate)
        else:
            clusters.append([candidate])
    return clusters


def cluster_temporal_windows(
    windows: Sequence[Sequence[float]],
    *,
    iou_threshold: float,
    output: str,
) -> list[list[float]]:
    """Collapse complete-link clusters to a representative or fused boundary."""
    if output not in {"representative", "fusion"}:
        raise ValueError(f"Unsupported cluster output: {output}")

    collapsed: list[list[float]] = []
    for cluster in complete_link_clusters(windows, iou_threshold):
        representative = max(
            enumerate(cluster), key=lambda item: (float(item[1][2]), -item[0])
        )[1]
        if output == "representative" or len(cluster) == 1:
            collapsed.append(list(representative))
            continue

        weights = np.asarray([max(float(window[2]), 0.0) for window in cluster], dtype=np.float64)
        if float(weights.sum()) <= 0:
            weights = np.ones(len(cluster), dtype=np.float64)
        starts = np.asarray([float(window[0]) for window in cluster], dtype=np.float64)
        ends = np.asarray([float(window[1]) for window in cluster], dtype=np.float64)
        collapsed.append([
            float(np.average(starts, weights=weights)),
            float(np.average(ends, weights=weights)),
            float(representative[2]),
        ])
    return _score_sorted(collapsed)


def apply_dedup_spec(
    windows: Sequence[Sequence[float]],
    spec: DedupSpec,
) -> list[list[float]]:
    if spec.method == "none":
        return [list(window) for window in windows]
    if spec.method == "hard_nms":
        return hard_temporal_nms(windows, float(spec.iou_threshold))
    if spec.method == "diou_nms":
        return diou_temporal_nms(windows, float(spec.iou_threshold))
    if spec.method == "soft_nms_linear":
        return soft_temporal_nms(
            windows,
            mode="linear",
            iou_threshold=float(spec.iou_threshold),
            score_floor=float(spec.score_floor),
        )
    if spec.method == "soft_nms_gaussian":
        return soft_temporal_nms(
            windows,
            mode="gaussian",
            sigma=float(spec.sigma),
            score_floor=float(spec.score_floor),
        )
    if spec.method == "cluster_representative":
        return cluster_temporal_windows(
            windows,
            iou_threshold=float(spec.iou_threshold),
            output="representative",
        )
    if spec.method == "cluster_fusion":
        return cluster_temporal_windows(
            windows,
            iou_threshold=float(spec.iou_threshold),
            output="fusion",
        )
    raise ValueError(f"Unknown dedup method: {spec.method}")


def _has_split_token(path: Path, token: str) -> bool:
    pattern = re.compile(rf"(^|[^a-z]){re.escape(token)}([^a-z]|$)")
    return any(pattern.search(part.lower()) for part in path.parts)


def assert_validation_input_path(path: str | Path, *, role: str) -> Path:
    """Reject test-labelled paths and require an explicit val/validation token."""
    candidate = Path(path)
    if _has_split_token(candidate, "test"):
        raise ValueError(f"Refusing to read test-labelled {role}: {candidate}")
    has_val = _has_split_token(candidate, "val") or _has_split_token(candidate, "validation")
    if not has_val:
        raise ValueError(
            f"Validation-only guard: {role} path must contain a standalone val/validation token: {candidate}"
        )
    return candidate


def _validate_split_metadata(rows: Sequence[dict[str, Any]], *, role: str) -> None:
    for index, row in enumerate(rows):
        split = row.get("split")
        if split is None:
            continue
        if str(split).lower() not in {"val", "valid", "validation"}:
            raise ValueError(f"{role} row {index} declares non-validation split={split!r}")


def _normalize_prediction_rows(rows: Sequence[Any]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    seen_qids: set[Any] = set()
    for row_index, raw in enumerate(rows):
        if not isinstance(raw, dict) or "qid" not in raw:
            raise ValueError(f"Prediction row {row_index} must be an object with qid")
        qid = raw["qid"]
        if qid in seen_qids:
            raise ValueError(f"Duplicate prediction qid={qid!r}")
        seen_qids.add(qid)

        windows = raw.get("pred_relevant_windows", [])
        if not isinstance(windows, list):
            raise ValueError(f"qid={qid!r}: pred_relevant_windows must be a list")
        clean_windows: list[list[float]] = []
        for window_index, window in enumerate(windows):
            if not isinstance(window, (list, tuple)) or len(window) < 2:
                raise ValueError(f"qid={qid!r} window {window_index}: expected [start,end,score]")
            try:
                start = float(window[0])
                end = float(window[1])
                score = float(window[2]) if len(window) >= 3 else 1.0
            except (TypeError, ValueError) as exc:
                raise ValueError(f"qid={qid!r} window {window_index}: non-numeric value") from exc
            if not all(math.isfinite(value) for value in (start, end, score)):
                raise ValueError(f"qid={qid!r} window {window_index}: non-finite value")
            if end <= start:
                raise ValueError(f"qid={qid!r} window {window_index}: end must be greater than start")
            clean_windows.append([start, end, score])

        row = copy.deepcopy(raw)
        row["pred_relevant_windows"] = clean_windows
        normalized.append(row)
    return normalized


def _validate_qid_coverage(
    submission: Sequence[dict[str, Any]],
    ground_truth: Sequence[dict[str, Any]],
) -> None:
    prediction_qids = {row["qid"] for row in submission}
    gt_qids = {row["qid"] for row in ground_truth}
    if len(gt_qids) != len(ground_truth):
        raise ValueError("Ground truth contains duplicate qids")
    if prediction_qids != gt_qids:
        missing = sorted(gt_qids - prediction_qids, key=str)[:10]
        extra = sorted(prediction_qids - gt_qids, key=str)[:10]
        raise ValueError(
            "Prediction qid coverage differs from validation GT: "
            f"prediction={len(prediction_qids)}, gt={len(gt_qids)}, "
            f"missing(sample)={missing}, extra(sample)={extra}"
        )


def audit_prediction_schema(submission: Sequence[dict[str, Any]]) -> dict[str, Any]:
    fields = sorted({key for row in submission for key in row})
    candidate_embedding_fields = sorted({
        key
        for key in fields
        if any(token in key.lower() for token in ("query_embed", "query_feature", "decoder_feature"))
    })
    return {
        "num_rows": len(submission),
        "top_level_fields": fields,
        "num_windows": sum(len(row.get("pred_relevant_windows", [])) for row in submission),
        "explicit_existence_coverage": sum("pred_exist_score" in row for row in submission),
        "count_prediction_coverage": sum("pred_count" in row for row in submission),
        "count_probability_coverage": sum("pred_count_probs" in row for row in submission),
        "candidate_embedding_fields": candidate_embedding_fields,
        "supports_geometry_score_dedup": True,
        "supports_candidate_semantic_dedup": bool(candidate_embedding_fields),
        "semantic_dedup_gap": (
            None
            if candidate_embedding_fields
            else "Candidate-level decoder embeddings/event IDs are absent; only temporal geometry and scores can be audited."
        ),
    }


def duplicate_diagnostics(
    submission: Sequence[dict[str, Any]],
    thresholds: Sequence[float] = (0.5, 0.7, 0.9),
) -> dict[str, Any]:
    counts = [len(row.get("pred_relevant_windows", [])) for row in submission]
    diagnostics: dict[str, Any] = {
        "num_queries": len(submission),
        "total_windows": int(sum(counts)),
        "mean_windows_per_query": round(float(np.mean(counts)) if counts else 0.0, 6),
        "median_windows_per_query": round(float(np.median(counts)) if counts else 0.0, 6),
        "max_windows_per_query": max(counts, default=0),
        "pairwise_overlap": {},
    }
    for threshold in thresholds:
        total_pairs = 0
        duplicate_pairs = 0
        queries_with_duplicate = 0
        for row in submission:
            windows = row.get("pred_relevant_windows", [])
            query_has_duplicate = False
            for first_index in range(len(windows)):
                for second_index in range(first_index + 1, len(windows)):
                    total_pairs += 1
                    if temporal_iou(windows[first_index], windows[second_index]) > threshold:
                        duplicate_pairs += 1
                        query_has_duplicate = True
            queries_with_duplicate += int(query_has_duplicate)
        diagnostics["pairwise_overlap"][str(threshold)] = {
            "pairs_above_threshold": duplicate_pairs,
            "total_pairs": total_pairs,
            "pair_rate": round(duplicate_pairs / total_pairs, 6) if total_pairs else 0.0,
            "queries_with_pair": queries_with_duplicate,
            "query_rate": round(queries_with_duplicate / len(submission), 6) if submission else 0.0,
        }
    return diagnostics


def _float_tag(value: float) -> str:
    return format(float(value), ".6g").replace("-", "m").replace(".", "p").replace("+", "")


def spec_name(spec: DedupSpec) -> str:
    pieces = [spec.method]
    if spec.iou_threshold is not None:
        pieces.append(f"iou{_float_tag(spec.iou_threshold)}")
    if spec.sigma is not None:
        pieces.append(f"sigma{_float_tag(spec.sigma)}")
    if spec.score_floor is not None:
        pieces.append(f"floor{_float_tag(spec.score_floor)}")
    if spec.cluster_linkage is not None:
        pieces.append(spec.cluster_linkage)
    return "__".join(pieces)


def build_specs(args: argparse.Namespace) -> list[DedupSpec]:
    specs: list[DedupSpec] = []
    # A direct Top-K baseline is mandatory for every selection budget.
    methods = ["none", *[method for method in args.methods if method != "none"]]
    for method in methods:
        if method == "none":
            specs.append(DedupSpec(method="none"))
        elif method in {"hard_nms", "diou_nms", "soft_nms_linear", "cluster_representative", "cluster_fusion"}:
            for threshold in args.iou_thresholds:
                specs.append(DedupSpec(
                    method=method,
                    iou_threshold=float(threshold),
                    score_floor=float(args.soft_score_floor) if method == "soft_nms_linear" else None,
                    cluster_linkage="complete" if method.startswith("cluster_") else None,
                ))
        elif method == "soft_nms_gaussian":
            for sigma in args.soft_sigmas:
                specs.append(DedupSpec(
                    method=method,
                    sigma=float(sigma),
                    score_floor=float(args.soft_score_floor),
                ))
        else:
            raise ValueError(f"Unknown method: {method}")
    names = [spec_name(spec) for spec in specs]
    if len(names) != len(set(names)):
        raise ValueError("Dedup grid contains duplicate method specifications")
    return specs


def process_submission(
    submission: Sequence[dict[str, Any]],
    spec: DedupSpec,
) -> list[dict[str, Any]]:
    processed: list[dict[str, Any]] = []
    for source in submission:
        row = copy.deepcopy(source)
        row["pred_relevant_windows"] = apply_dedup_spec(
            source.get("pred_relevant_windows", []), spec
        )
        processed.append(row)
    return processed


def build_selection_budgets(values: Sequence[str]) -> list[int | str]:
    budgets: list[int | str] = []
    for value in values:
        if value == "predicted_count":
            budgets.append(value)
            continue
        try:
            fixed_k = int(value)
        except ValueError as exc:
            raise ValueError(
                f"Selection budget must be a positive integer or predicted_count, got {value!r}"
            ) from exc
        if fixed_k <= 0:
            raise ValueError(f"Fixed selection budget must be positive, got {fixed_k}")
        budgets.append(fixed_k)
    if not {1, 3, 5}.issubset(set(budget for budget in budgets if isinstance(budget, int))):
        raise ValueError("Fair dedup ablation requires fixed Top-K budgets 1, 3, and 5")
    if "predicted_count" not in budgets:
        raise ValueError("Fair dedup ablation requires the predicted_count budget")
    if len(budgets) != len(set(budgets)):
        raise ValueError("Selection budgets contain duplicates")
    return budgets


def selection_budget_name(budget: int | str) -> str:
    return f"topk{budget}" if isinstance(budget, int) else "predicted_count_topk"


def _requested_window_count(row: dict[str, Any], budget: int | str) -> int:
    available = len(row.get("pred_relevant_windows", []))
    if isinstance(budget, int):
        return min(budget, available)
    if "pred_count" not in row:
        raise ValueError(
            f"qid={row.get('qid')!r} lacks pred_count required by predicted_count budget"
        )
    try:
        predicted_count = int(row["pred_count"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"qid={row.get('qid')!r} has invalid pred_count={row['pred_count']!r}") from exc
    # Class 4 denotes 4+ in the current counter.  Treating it as a lower-bound
    # Top-4 budget is explicit and deterministic; no score threshold is added.
    predicted_count = max(0, min(predicted_count, 4))
    return min(predicted_count, available)


def apply_selection_budget(
    ranked_submission: Sequence[dict[str, Any]],
    budget: int | str,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for source in ranked_submission:
        row = copy.deepcopy(source)
        requested = _requested_window_count(source, budget)
        row["pred_relevant_windows"] = row.get("pred_relevant_windows", [])[:requested]
        selected.append(row)
    return selected


def selection_diagnostics(
    source_submission: Sequence[dict[str, Any]],
    selected_submission: Sequence[dict[str, Any]],
    budget: int | str,
) -> dict[str, Any]:
    selected_by_qid = {row["qid"]: row for row in selected_submission}
    requested_total = 0
    selected_total = 0
    underfilled_qids: list[Any] = []
    for source in source_submission:
        requested = _requested_window_count(source, budget)
        selected_count = len(selected_by_qid[source["qid"]].get("pred_relevant_windows", []))
        requested_total += requested
        selected_total += selected_count
        if selected_count < requested:
            underfilled_qids.append(source["qid"])
    return {
        "budget": budget,
        "budget_semantics": (
            "at_most_fixed_k_after_dedup"
            if isinstance(budget, int)
            else "pred_count clipped to [0,4]; class 4+ uses its Top-4 lower bound"
        ),
        "requested_total_from_source": requested_total,
        "selected_total": selected_total,
        "underfilled_queries": len(underfilled_qids),
        "underfilled_qid_sample": underfilled_qids[:20],
    }


def brief_delta(
    candidate: dict[str, Any],
    direct_topk: dict[str, Any],
) -> dict[str, float]:
    delta: dict[str, float] = {}
    for key, value in candidate.items():
        baseline_value = direct_topk.get(key)
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and isinstance(baseline_value, (int, float))
            and not isinstance(baseline_value, bool)
        ):
            delta[key] = round(float(value) - float(baseline_value), 6)
    return delta


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def run_ablation(args: argparse.Namespace) -> dict[str, Any]:
    prediction_path = assert_validation_input_path(args.prediction_path, role="prediction")
    gt_path = assert_validation_input_path(args.gt_path, role="ground truth")
    if not prediction_path.is_file():
        raise FileNotFoundError(prediction_path)
    if not gt_path.is_file():
        raise FileNotFoundError(gt_path)

    raw_submission = load_jsonl(str(prediction_path))
    raw_gt = load_jsonl(str(gt_path))
    _validate_split_metadata(raw_submission, role="prediction")
    _validate_split_metadata(raw_gt, role="ground truth")

    submission = _normalize_prediction_rows(raw_submission)
    ts_config = load_ts_window_cfg(args.gt_ts_window_cfg)
    ground_truth, gt_stats = normalize_ground_truth(raw_gt, ts_config, drop_empty_gt=False)
    _validate_qid_coverage(submission, ground_truth)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = build_specs(args)
    selection_budgets = build_selection_budgets(args.selection_budgets)
    source_diagnostics = duplicate_diagnostics(submission, args.diagnostic_iou_thresholds)
    schema_audit = audit_prediction_schema(submission)

    summary: dict[str, Any] = {
        "protocol": {
            "split": "validation",
            "test_labels_read": False,
            "prediction_path": str(prediction_path),
            "gt_path": str(gt_path),
            "gt_normalization": gt_stats,
            "k_list": list(args.k_list),
            "max_pred_windows": args.max_pred_windows,
            "cls_thresholds": list(args.cls_thresholds),
            "gmiou_cls_threshold": args.gmiou_cls_threshold,
            "selection_budgets": selection_budgets,
            "note": (
                "Existence/count fields and the existence gate are preserved. Each dedup method is "
                "compared with direct Top-K under the same fixed/predicted-count output budget."
            ),
        },
        "prediction_schema_audit": schema_audit,
        "source_duplicate_diagnostics": source_diagnostics,
        "methods": [],
    }

    direct_brief_by_budget: dict[str, dict[str, Any]] = {}
    for spec in specs:
        dedup_name = spec_name(spec)
        ranked = process_submission(submission, spec)
        before_selection = duplicate_diagnostics(ranked, args.diagnostic_iou_thresholds)
        for budget in selection_budgets:
            budget_name = selection_budget_name(budget)
            name = f"{dedup_name}__{budget_name}"
            processed = apply_selection_budget(ranked, budget)
            diagnostics = duplicate_diagnostics(processed, args.diagnostic_iou_thresholds)
            diagnostics["dedup_before_selection"] = before_selection
            diagnostics["selection"] = selection_diagnostics(submission, processed, budget)
            diagnostics["windows_removed_vs_source"] = (
                source_diagnostics["total_windows"] - diagnostics["total_windows"]
            )
            metrics = evaluate_gmr(
                processed,
                ground_truth,
                k_list=tuple(args.k_list),
                max_pred_windows=args.max_pred_windows,
                cls_thresholds=tuple(args.cls_thresholds),
                gmiou_cls_threshold=args.gmiou_cls_threshold,
                iou_thds=DEFAULT_IOU_THRESHOLDS,
                map_num_workers=args.map_num_workers,
                verbose=False,
            )
            brief = dict(metrics["brief"])
            if spec.method == "none":
                direct_brief_by_budget[budget_name] = brief
            direct_brief = direct_brief_by_budget.get(budget_name)
            if direct_brief is None:
                raise AssertionError(f"Missing mandatory direct Top-K baseline for {budget_name}")
            delta_vs_direct = brief_delta(brief, direct_brief)

            prediction_output = output_dir / f"{name}.predictions.jsonl"
            metrics_output = output_dir / f"{name}.metrics.json"
            _write_jsonl(prediction_output, processed)
            metrics_payload = dict(metrics)
            metrics_payload["dedup_ablation"] = {
                "split": "validation",
                "spec": asdict(spec),
                "selection_budget": budget,
                "direct_topk_baseline": f"none__{budget_name}",
                "delta_vs_direct_topk": delta_vs_direct,
                "existence_score_or_gate_changed": False,
                "diagnostics": diagnostics,
                "source_prediction": str(prediction_path),
            }
            _write_json(metrics_output, metrics_payload)

            summary["methods"].append({
                "name": name,
                "spec": asdict(spec),
                "selection_budget": budget,
                "direct_topk_baseline": f"none__{budget_name}",
                "delta_vs_direct_topk": delta_vs_direct,
                "prediction_output": str(prediction_output),
                "metrics_output": str(metrics_output),
                "brief": brief,
                "diagnostics": diagnostics,
            })

    summary_path = output_dir / "dedup_ablation_summary.json"
    summary["summary_path"] = str(summary_path)
    _write_json(summary_path, summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run validation-only temporal deduplication ablations.",
        allow_abbrev=False,
    )
    parser.add_argument("--prediction-path", required=True, help="Validation prediction JSONL")
    parser.add_argument("--gt-path", required=True, help="Validation ground-truth JSONL")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gt-ts-window-cfg", default=None)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--iou-thresholds", nargs="+", type=float, default=[0.5, 0.7, 0.9])
    parser.add_argument("--soft-sigmas", nargs="+", type=float, default=[0.5])
    parser.add_argument(
        "--soft-score-floor",
        type=float,
        default=0.0,
        help="Default 0 preserves the same candidate budget and only re-ranks Soft-NMS outputs.",
    )
    parser.add_argument(
        "--selection-budgets",
        nargs="+",
        default=["1", "3", "5", "predicted_count"],
        help="Required fair budgets: 1 3 5 predicted_count",
    )
    parser.add_argument("--diagnostic-iou-thresholds", nargs="+", type=float, default=[0.5, 0.7, 0.9])
    parser.add_argument("--k-list", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument("--max-pred-windows", type=int, default=10)
    parser.add_argument("--cls-thresholds", nargs="+", type=float, default=[0.4, 0.6])
    parser.add_argument("--gmiou-cls-threshold", type=float, default=0.4)
    parser.add_argument("--map-num-workers", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    summary = run_ablation(args)
    compact = {
        row["name"]: {
            "mAP": row["brief"].get("mAP"),
            "G-mIoU@3": row["brief"].get("G-mIoU@3"),
            "mR+@3": row["brief"].get("mR+@3"),
            "windows": row["diagnostics"]["total_windows"],
        }
        for row in summary["methods"]
    }
    print(json.dumps(compact, indent=2, ensure_ascii=False))
    print(f"Saved summary -> {summary['summary_path']}")


if __name__ == "__main__":
    main()
