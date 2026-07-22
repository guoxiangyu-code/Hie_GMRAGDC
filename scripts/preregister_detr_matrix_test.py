#!/usr/bin/env python3
"""Freeze a validation-selected, one-shot multi-backbone DETR test matrix.

Schema version 3 deliberately describes execution as a DAG.  Checkpoints,
the test annotation, upstream step outputs, and output locations are bound to
the exact flags consumed by an allow-listed evaluator.  The companion runner
``scripts/run_preregistered_detr_matrix_test.py`` is the only supported way to
execute the resulting manifest.

The freezer hashes the test annotation bytewise; it never evaluates or parses
its labels.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import datetime as dt
import hashlib
import importlib
import io
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(REPO_ROOT):
    sys.path.insert(0, str(REPO_ROOT))

SCHEMA_VERSION = 3
PROTOCOL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
ROLE_NAMES = {"anchor", "diagnostic", "candidate"}
METRIC_NAMES = ("mAP", "G-mIoU@3", "mR+@5")
ALLOWED_ENV_KEYS = {
    "CUDA_VISIBLE_DEVICES",
    "PYTHONHASHSEED",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
}

DEFAULT_TRUSTED_REGISTRY_ROOT = REPO_ROOT / "artifacts" / "blind_test_registry"

# The roster is a security boundary, not presentation metadata.  A backbone
# may only execute modules named here, and every module is rebound to its
# repository source file when the manifest is rebuilt by the runner.
BACKBONE_ROSTER: dict[str, dict[str, Any]] = {
    "moment_detr": {
        "modules": {
            "training.moment_detr_gmr.evaluate",
            "scripts.fuse_gmr_heads",
            "eval.eval_main",
            "scripts.diagnose_gmr_groups",
        },
    },
    "qd_detr": {
        "modules": {
            "methods.qd_detr_gmr.evaluate",
            "eval.eval_main",
            "scripts.diagnose_gmr_groups",
        },
    },
    "cg_detr": {
        "modules": {
            "methods.cg_detr_gmr.evaluate",
            "eval.eval_main",
            "scripts.diagnose_gmr_groups",
        },
    },
    "eatr": {
        "modules": {
            "methods.eatr_gmr.evaluate",
            "eval.eval_main",
            "scripts.diagnose_gmr_groups",
        },
    },
}

PARSER_FACTORIES = {
    "training.moment_detr_gmr.evaluate": "build_parser",
    "methods.qd_detr_gmr.evaluate": "build_parser",
    "methods.cg_detr_gmr.evaluate": "build_parser",
    "methods.eatr_gmr.evaluate": "make_parser",
    "scripts.fuse_gmr_heads": "build_parser",
    "eval.eval_main": "build_parser",
    "scripts.diagnose_gmr_groups": "build_parser",
}

# These contracts are intentionally small.  A new evaluator must first state
# which flags own annotations, checkpoints, upstream data, and outputs.
COMMAND_CONTRACTS: dict[str, dict[str, Any]] = {
    "training.moment_detr_gmr.evaluate": {
        "kind": "predict",
        "annotation_flag": "--eval_path",
        "checkpoint_flags": {"--model_path"},
        "input_flags": set(),
        "output_flags": {"--results_dir"},
        "required_output_flags": {"--results_dir"},
        "produces_primary_metrics": False,
        "required_pairs": {"--split": "test"},
        "static_input_flags": {"--v_feat_dirs": 2, "--t_feat_dir": 1},
    },
    "methods.qd_detr_gmr.evaluate": {
        "kind": "evaluate",
        "annotation_flag": "--eval_annotation",
        "checkpoint_flags": {"--checkpoint"},
        "input_flags": set(),
        "output_flags": {"--submission_path", "--metrics_path"},
        "required_output_flags": {"--submission_path", "--metrics_path"},
        "produces_primary_metrics": True,
        "static_input_flags": {
            "--video_feature_dirs": 2, "--text_feature_dir": 1,
        },
    },
    "methods.cg_detr_gmr.evaluate": {
        "kind": "evaluate",
        "annotation_flag": "--eval_annotation",
        "checkpoint_flags": {"--checkpoint"},
        "input_flags": set(),
        "output_flags": {"--submission_path", "--metrics_path"},
        "required_output_flags": {"--submission_path", "--metrics_path"},
        "produces_primary_metrics": True,
        "static_input_flags": {
            "--video_feature_dirs": 2, "--text_feature_dir": 1,
        },
    },
    "methods.eatr_gmr.evaluate": {
        "kind": "evaluate",
        "annotation_flag": "--annotations",
        "checkpoint_flags": {"--checkpoint"},
        "input_flags": set(),
        "output_flags": {"--output-dir"},
        "required_output_flags": {"--output-dir"},
        "produces_primary_metrics": True,
        "static_input_flags": {
            "--slowfast-dir": 1, "--clip-dir": 1, "--text-dir": 1,
        },
    },
    "scripts.fuse_gmr_heads": {
        "kind": "compose_evaluate",
        "annotation_flag": "--ground-truth",
        "checkpoint_flags": set(),
        "input_flags": {"--localization", "--decision"},
        "required_input_flags": {"--localization", "--decision"},
        "output_flags": {"--output", "--manifest", "--metrics-output"},
        "required_output_flags": {"--output", "--manifest", "--metrics-output"},
        "produces_primary_metrics": True,
    },
    "eval.eval_main": {
        "kind": "evaluate",
        "annotation_flag": "--gt_path",
        "checkpoint_flags": set(),
        "input_flags": {"--submission_path"},
        "required_input_flags": {"--submission_path"},
        "output_flags": {"--save_path"},
        "required_output_flags": {"--save_path"},
        "produces_primary_metrics": True,
    },
    "scripts.diagnose_gmr_groups": {
        "kind": "diagnose",
        "annotation_flag": "--ground-truth",
        "checkpoint_flags": set(),
        "input_flags": {"--submission"},
        "required_input_flags": {"--submission"},
        "output_flags": {"--output"},
        "required_output_flags": {"--output"},
        "produces_primary_metrics": False,
    },
}

FORBIDDEN_FLAGS = {
    "--allow_partial_load",
    "--allow-partial-load",
    "--max_eval_samples",
    "--max-eval-samples",
}


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _lexical_absolute(path: str | Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _canonical_non_symlink(path: str | Path, label: str, *, must_exist: bool) -> Path:
    raw = Path(path)
    if not raw.is_absolute():
        raise ValueError(f"{label} must be an absolute path: {path}")
    lexical = _lexical_absolute(raw)
    try:
        resolved = raw.resolve(strict=must_exist)
    except FileNotFoundError:
        raise FileNotFoundError(f"{label} does not exist: {raw}") from None
    if lexical != resolved:
        raise ValueError(
            f"{label} must be canonical and have no symlink ancestors: {path}"
        )
    return resolved


def artifact(path: str | Path) -> dict[str, Any]:
    raw = Path(path)
    lexical = _lexical_absolute(raw)
    resolved = raw.resolve()
    if lexical != resolved:
        raise FileNotFoundError(
            f"frozen artifact must have no symlink ancestors: {raw}"
        )
    if not resolved.is_file():
        raise FileNotFoundError(f"frozen artifact must be a regular non-symlink file: {resolved}")
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def verify_artifact(value: dict[str, Any]) -> None:
    current = artifact(value["path"])
    expected = {key: value[key] for key in ("path", "bytes", "sha256")}
    if current != expected:
        raise RuntimeError(
            f"frozen artifact changed: {value['path']} expected={expected} current={current}"
        )


def load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def brief(path: str | Path) -> dict[str, Any]:
    value = load_json(path)
    result = value.get("brief", value)
    if not isinstance(result, dict):
        raise TypeError(f"metrics brief must be an object: {path}")
    return result


def _require_absolute(path: str | Path, label: str) -> Path:
    return _canonical_non_symlink(path, label, must_exist=False)


def ensure_pristine(path: str | Path) -> str:
    resolved = _canonical_non_symlink(
        path, "test output root", must_exist=False
    )
    if resolved.exists() and not resolved.is_dir():
        raise RuntimeError(f"expected test output root is not a directory: {resolved}")
    children = list(resolved.iterdir()) if resolved.exists() else []
    if children:
        raise RuntimeError(
            f"test output root is not pristine: {[str(child) for child in children[:5]]}"
        )
    return str(resolved)


def paths_overlap(first: Path, second: Path) -> bool:
    return first == second or first in second.parents or second in first.parents


def _json_safe(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"argparse produced a non-JSON value: {type(value).__name__}")


def _module_source(module: str, repo_root: Path) -> Path:
    expected = repo_root.joinpath(*module.split(".")).with_suffix(".py")
    return _canonical_non_symlink(
        expected, f"trusted module source for {module}", must_exist=True
    )


def _parser_for_module(module: str, repo_root: Path) -> argparse.ArgumentParser:
    factory_name = PARSER_FACTORIES.get(module)
    if factory_name is None:
        raise ValueError(f"no trusted parser factory for module {module!r}")
    imported = importlib.import_module(module)
    imported_path = _canonical_non_symlink(
        Path(imported.__file__ or ""), f"imported module {module}", must_exist=True
    )
    expected_path = _module_source(module, repo_root)
    if imported_path != expected_path:
        raise RuntimeError(
            f"module {module!r} resolved outside the trusted repository: {imported_path}"
        )
    parser = getattr(imported, factory_name)()
    if not isinstance(parser, argparse.ArgumentParser):
        raise TypeError(f"{module}.{factory_name} did not return ArgumentParser")
    parser.allow_abbrev = False
    return parser


def canonicalize_argv(
    argv: list[str], *, module: str, repo_root: Path,
) -> dict[str, Any]:
    """Parse an argv with exact spellings and reject duplicate destinations.

    ``argparse`` normally accepts both abbreviations and repeated destinations
    (last value wins).  Neither behavior is safe in a frozen protocol.  This
    pre-scan admits only the parser's canonical long spelling, then records the
    complete parsed namespace so the runner can reproduce the same semantics.
    """
    parser = _parser_for_module(module, repo_root)
    actions = [action for action in parser._actions if action.dest != "help"]
    option_to_action = {
        option: action for action in actions for option in action.option_strings
    }
    canonical_options: set[str] = set()
    for action in actions:
        long_options = [
            option for option in action.option_strings if option.startswith("--")
        ]
        if not long_options:
            continue
        positive = next(
            (option for option in long_options if not option.startswith("--no-")),
            long_options[0],
        )
        canonical_options.add(positive)
        if isinstance(action, argparse.BooleanOptionalAction):
            negative = next(
                (
                    option for option in long_options
                    if option == "--no-" + positive[2:]
                ),
                None,
            )
            if negative is not None:
                canonical_options.add(negative)

    seen_destinations: dict[str, str] = {}
    explicit_destinations: list[str] = []
    for token in argv[3:]:
        if token.startswith("--"):
            if "=" in token:
                raise ValueError(
                    f"module {module!r} argv must use separated --flag value syntax"
                )
            action = option_to_action.get(token)
            if action is None:
                raise ValueError(f"module {module!r} argv has unknown option {token!r}")
            if token not in canonical_options:
                raise ValueError(
                    f"module {module!r} argv uses non-canonical option alias {token!r}"
                )
            previous = seen_destinations.get(action.dest)
            if previous is not None:
                raise ValueError(
                    f"module {module!r} argv repeats destination {action.dest!r} "
                    f"through {previous!r} and {token!r}"
                )
            seen_destinations[action.dest] = token
            explicit_destinations.append(action.dest)
        elif token.startswith("-") and not re.fullmatch(
            r"-[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?", token
        ):
            raise ValueError(
                f"module {module!r} argv may not use short/unknown option {token!r}"
            )

    try:
        with contextlib.redirect_stderr(io.StringIO()):
            namespace = parser.parse_args(argv[3:])
    except SystemExit as error:
        raise ValueError(f"module {module!r} argv fails its exact parser") from error
    parsed = {
        key: _json_safe(value) for key, value in sorted(vars(namespace).items())
    }
    return {
        "arguments": parsed,
        "explicit_destinations": sorted(explicit_destinations),
        "destination_options": dict(sorted(seen_destinations.items())),
    }


def _validate_metric_brief(metrics: dict[str, Any], label: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for name in METRIC_NAMES:
        if name not in metrics:
            raise ValueError(f"{label} is missing required metric {name!r}")
        try:
            number = float(metrics[name])
        except (TypeError, ValueError) as error:
            raise ValueError(f"{label} metric {name!r} is not numeric") from error
        if not math.isfinite(number) or not 0.0 <= number <= 100.0:
            raise ValueError(f"{label} metric {name!r} must be finite and in [0, 100]")
        result[name] = number
    return result


def _diagnostic_values(diagnostics: dict[str, Any], label: str) -> dict[str, float]:
    groups = diagnostics.get("groups")
    if not isinstance(groups, dict):
        raise ValueError(f"{label} diagnostics need a groups object")
    result = {}
    for group_name in ("single", "multi"):
        group = groups.get(group_name)
        if not isinstance(group, dict):
            raise ValueError(f"{label} diagnostics need group {group_name!r}")
        support = group.get("support")
        if not isinstance(support, int) or isinstance(support, bool) or support <= 0:
            raise ValueError(f"{label} diagnostics {group_name}.support must be positive")
        try:
            acceptance = float(group["acceptance_rate"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"{label} diagnostics {group_name}.acceptance_rate must be numeric"
            ) from error
        if not math.isfinite(acceptance) or not 0.0 <= acceptance <= 100.0:
            raise ValueError(
                f"{label} diagnostics {group_name}.acceptance_rate must be in [0, 100]"
            )
        result[f"{group_name}_acceptance"] = acceptance
    try:
        balanced = float(diagnostics["balanced_G"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{label} diagnostics balanced_G must be numeric") from error
    if not math.isfinite(balanced) or not 0.0 <= balanced <= 100.0:
        raise ValueError(f"{label} diagnostics balanced_G must be in [0, 100]")
    result["balanced_G"] = balanced
    return result


def candidate_gates(candidate: dict[str, Any], reference: dict[str, Any],
                    diagnostics: dict[str, Any], *, label: str = "candidate") -> dict[str, bool]:
    candidate_values = _validate_metric_brief(candidate, label)
    reference_values = _validate_metric_brief(reference, f"{label} reference")
    diagnostic_values = _diagnostic_values(diagnostics, label)
    return {
        "mAP_improves": candidate_values["mAP"] > reference_values["mAP"],
        "G-mIoU@3_improves": (
            candidate_values["G-mIoU@3"] > reference_values["G-mIoU@3"]
        ),
        "mR+@5_guard": (
            candidate_values["mR+@5"] >= reference_values["mR+@5"] - 0.5
        ),
        "balanced_G_above_all_empty": diagnostic_values["balanced_G"] > 50.0,
        "single_acceptance_positive": diagnostic_values["single_acceptance"] > 0.0,
        "multi_acceptance_positive": diagnostic_values["multi_acceptance"] > 0.0,
    }


def _artifact_identity(value: dict[str, Any], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"path", "bytes", "sha256"}:
        raise ValueError(f"{label} must be an exact artifact identity")
    path = _require_absolute(value["path"], f"{label} path")
    if not isinstance(value["bytes"], int) or isinstance(value["bytes"], bool):
        raise ValueError(f"{label} bytes must be an integer")
    if not isinstance(value["sha256"], str) or not re.fullmatch(
        r"[0-9a-f]{64}", value["sha256"]
    ):
        raise ValueError(f"{label} sha256 must be lowercase hexadecimal")
    return {"path": str(path), "bytes": value["bytes"], "sha256": value["sha256"]}


def validate_checkpoint_validation_provenance(
    value: dict[str, Any], *, checkpoint_roles: dict[str, dict[str, Any]],
    validation_metrics: dict[str, Any], validation_annotations: dict[str, Any],
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError(f"{label} provenance schema_version must be 1")
    if value.get("selection_split") != "validation":
        raise ValueError(f"{label} provenance selection_split must be validation")
    seed = value.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError(f"{label} provenance seed must be a non-negative integer")
    claimed_roles = value.get("checkpoint_roles")
    if not isinstance(claimed_roles, dict) or set(claimed_roles) != set(checkpoint_roles):
        raise ValueError(f"{label} provenance checkpoint roles do not close")
    for role, expected in checkpoint_roles.items():
        claimed = _artifact_identity(
            claimed_roles[role], f"{label} provenance checkpoint {role}"
        )
        if claimed != expected:
            raise ValueError(
                f"{label} provenance checkpoint role {role!r} does not match"
            )
    claimed_metrics = _artifact_identity(
        value.get("validation_metrics"), f"{label} provenance validation metrics"
    )
    if claimed_metrics != validation_metrics:
        raise ValueError(f"{label} provenance validation metrics do not match")
    claimed_annotations = _artifact_identity(
        value.get("validation_annotations"),
        f"{label} provenance validation annotations",
    )
    if claimed_annotations != validation_annotations:
        raise ValueError(f"{label} provenance validation annotations do not match")
    return {
        "schema_version": 1,
        "selection_split": "validation",
        "seed": seed,
        "checkpoint_roles": claimed_roles,
        "validation_metrics": claimed_metrics,
        "validation_annotations": claimed_annotations,
    }


def validate_three_seed_summary(
    value: dict[str, Any], *, label: str,
) -> tuple[dict[str, Any], dict[str, bool]]:
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError(f"{label} three-seed schema_version must be 1")
    rows = value.get("seeds")
    if not isinstance(rows, list) or len(rows) != 3:
        raise ValueError(f"{label} three-seed summary must contain exactly three seeds")
    seeds: set[int] = set()
    deltas = {"mAP": [], "G-mIoU@3": []}
    normalized_rows = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != {"seed", "candidate", "reference"}:
            raise ValueError(f"{label} three-seed row {index} has an invalid schema")
        seed = row["seed"]
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0 or seed in seeds:
            raise ValueError(f"{label} three-seed seeds must be unique non-negative integers")
        seeds.add(seed)
        normalized_metrics: dict[str, dict[str, float]] = {}
        for role in ("candidate", "reference"):
            metrics = row[role]
            if not isinstance(metrics, dict) or set(metrics) != {"mAP", "G-mIoU@3"}:
                raise ValueError(
                    f"{label} three-seed row {index} {role} needs mAP and G-mIoU@3"
                )
            normalized_metrics[role] = {}
            for metric in ("mAP", "G-mIoU@3"):
                try:
                    number = float(metrics[metric])
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"{label} three-seed {role} {metric} is not numeric"
                    ) from error
                if not math.isfinite(number) or not 0.0 <= number <= 100.0:
                    raise ValueError(
                        f"{label} three-seed {role} {metric} must be finite in [0, 100]"
                    )
                normalized_metrics[role][metric] = number
        for metric in deltas:
            deltas[metric].append(
                normalized_metrics["candidate"][metric]
                - normalized_metrics["reference"][metric]
            )
        normalized_rows.append({"seed": seed, **normalized_metrics})

    summary = value.get("summary")
    if not isinstance(summary, dict) or set(summary) != set(deltas):
        raise ValueError(f"{label} three-seed summary metrics are invalid")
    normalized_summary = {}
    gates = {}
    for metric, values in deltas.items():
        claimed = summary[metric]
        if not isinstance(claimed, dict) or set(claimed) != {
            "mean_delta", "all_seeds_improve",
        }:
            raise ValueError(f"{label} three-seed {metric} aggregate schema is invalid")
        mean_delta = sum(values) / 3.0
        all_improve = all(delta > 0.0 for delta in values)
        try:
            claimed_mean = float(claimed["mean_delta"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"{label} three-seed {metric} mean_delta is invalid") from error
        if not math.isfinite(claimed_mean) or not math.isclose(
            claimed_mean, mean_delta, rel_tol=0.0, abs_tol=1e-9
        ):
            raise ValueError(f"{label} three-seed {metric} mean_delta is inconsistent")
        if not isinstance(claimed["all_seeds_improve"], bool) or (
            claimed["all_seeds_improve"] != all_improve
        ):
            raise ValueError(
                f"{label} three-seed {metric} all_seeds_improve is inconsistent"
            )
        normalized_summary[metric] = {
            "mean_delta": mean_delta,
            "all_seeds_improve": all_improve,
        }
        gates[f"three_seed_{metric}_mean_improves"] = mean_delta > 0.0
        gates[f"three_seed_{metric}_all_improve"] = all_improve
    return {
        "schema_version": 1,
        "seeds": normalized_rows,
        "summary": normalized_summary,
    }, gates


def source_artifacts(paths: list[str]) -> list[dict[str, Any]]:
    result = []
    seen = set()
    for value in paths:
        path = Path(value)
        children = (
            sorted(
                child for child in path.rglob("*")
                if child.is_file() and not child.is_symlink()
                and "__pycache__" not in child.parts
            )
            if path.is_dir() else [path]
        )
        if not children:
            raise ValueError(f"source path contains no files: {path}")
        for child in children:
            resolved = str(child.resolve())
            if resolved in seen:
                continue
            seen.add(resolved)
            result.append(artifact(child))
    return result


def source_inventory(path: str | Path) -> dict[str, Any]:
    raw = Path(path)
    resolved = raw.resolve()
    if _lexical_absolute(raw) != resolved:
        raise ValueError(f"source path may not have symlink ancestors: {raw}")
    if resolved.is_file():
        frozen = artifact(resolved)
        return {
            "kind": "file",
            "path": frozen["path"],
            "bytes": frozen["bytes"],
            "sha256": frozen["sha256"],
        }
    if not resolved.is_dir():
        raise FileNotFoundError(resolved)
    files = []
    for child in sorted(resolved.rglob("*")):
        if "__pycache__" in child.parts:
            continue
        if child.is_symlink():
            raise ValueError(f"source directory contains a symlink: {child}")
        if child.is_file():
            frozen = artifact(child)
            files.append({
                "relative_path": str(child.relative_to(resolved)),
                "bytes": frozen["bytes"],
                "sha256": frozen["sha256"],
            })
    if not files:
        raise ValueError(f"source directory contains no files: {resolved}")
    return {
        "kind": "directory",
        "path": str(resolved),
        "files": files,
        "sha256": sha256_bytes(_canonical_json(files)),
    }


def verify_source_inventory(value: dict[str, Any]) -> None:
    current = source_inventory(value["path"])
    if current != value:
        raise RuntimeError(
            f"frozen source inventory changed: {value['path']}"
        )


def _normalize_static_inputs(entry: dict[str, Any], *, label: str) -> tuple[
    dict[str, dict[str, Any]], dict[str, str]
]:
    raw_inputs = entry.get("static_inputs")
    if not isinstance(raw_inputs, dict) or not raw_inputs:
        raise ValueError(f"{label} needs a non-empty static_inputs object")
    frozen: dict[str, dict[str, Any]] = {}
    paths: dict[str, str] = {}
    seen_paths: set[Path] = set()
    for role, value in raw_inputs.items():
        if not isinstance(role, str) or not role or not isinstance(value, str):
            raise ValueError(f"{label} has an invalid static input role")
        resolved = _canonical_non_symlink(
            value, f"{label} static input {role}", must_exist=True
        )
        if resolved in seen_paths:
            raise ValueError(f"{label} static input paths must be distinct: {resolved}")
        seen_paths.add(resolved)
        inventory = source_inventory(resolved)
        frozen[role] = inventory
        paths[role] = inventory["path"]
    return frozen, paths


def _flag_value(argv: list[str], flag: str) -> str:
    positions = [index for index, token in enumerate(argv) if token == flag]
    if len(positions) != 1:
        raise ValueError(f"flag {flag!r} must occur exactly once in argv")
    index = positions[0]
    if index + 1 >= len(argv) or argv[index + 1].startswith("--"):
        raise ValueError(f"flag {flag!r} needs one explicit value")
    return argv[index + 1]


def _normalize_outputs(step: dict[str, Any], root: Path,
                       used_outputs: set[Path]) -> dict[str, str]:
    raw_outputs = step.get("outputs")
    if not isinstance(raw_outputs, dict) or not raw_outputs:
        raise ValueError(f"step {step.get('id')!r} needs a non-empty outputs object")
    outputs: dict[str, str] = {}
    for name, value in raw_outputs.items():
        if not isinstance(name, str) or not name or not isinstance(value, str):
            raise ValueError(f"step {step.get('id')!r} has invalid output declaration")
        path = _require_absolute(value, f"step {step.get('id')} output {name}")
        if not (path == root or root in path.parents):
            raise ValueError(f"step output escapes expected output root: {path}")
        if path == root:
            raise ValueError(f"step output must be a file below the output root: {path}")
        overlap = next(
            (existing for existing in used_outputs if paths_overlap(path, existing)),
            None,
        )
        if overlap is not None:
            raise ValueError(
                f"expected output paths overlap as parent/child: {path} / {overlap}"
            )
        used_outputs.add(path)
        outputs[name] = str(path)
    return outputs


def _normalize_environment(step: dict[str, Any]) -> dict[str, str]:
    environment = step.get("environment", {})
    if not isinstance(environment, dict):
        raise ValueError(f"step {step.get('id')!r} environment must be an object")
    result = {}
    for name, value in environment.items():
        if name not in ALLOWED_ENV_KEYS or not isinstance(value, str):
            raise ValueError(f"step {step.get('id')!r} has unsafe environment key {name!r}")
        result[name] = value
    return result


def _normalize_step(
    step: dict[str, Any], *, checkpoint_paths: dict[str, str], test_annotations: Path,
    output_root: Path, working_directory: Path, used_outputs: set[Path],
    backbone: str, static_input_paths: dict[str, str], repo_root: Path,
) -> dict[str, Any]:
    step_id = step.get("id")
    if not isinstance(step_id, str) or not step_id:
        raise ValueError("every execution step needs a non-empty string id")
    argv = step.get("argv")
    if not isinstance(argv, list) or len(argv) < 3 or not all(
        isinstance(value, str) and value for value in argv
    ):
        raise ValueError(f"step {step_id!r} argv must be a non-empty string list")
    executable = _canonical_non_symlink(
        argv[0], f"step {step_id} executable", must_exist=True
    )
    if not executable.is_file():
        raise ValueError(f"step {step_id!r} executable must be a regular file")
    if not executable.name.startswith("python"):
        raise ValueError(f"step {step_id!r} executable must be Python")
    if argv[1] != "-m":
        raise ValueError(f"step {step_id!r} must use an allow-listed python -m module")
    module = argv[2]
    contract = COMMAND_CONTRACTS.get(module)
    if contract is None:
        raise ValueError(f"step {step_id!r} uses unsupported module {module!r}")
    if module not in BACKBONE_ROSTER[backbone]["modules"]:
        raise ValueError(
            f"step {step_id!r} module {module!r} is not allowed for backbone {backbone!r}"
        )
    semantic_argv = canonicalize_argv(argv, module=module, repo_root=repo_root)
    explicit_destinations = set(semantic_argv["explicit_destinations"])
    if explicit_destinations & {"allow_partial_load", "max_eval_samples"}:
        raise ValueError(f"step {step_id!r} contains a forbidden evaluation flag")
    if module == "methods.qd_detr_gmr.evaluate":
        decoder_switches = sum(
            flag in argv
            for flag in ("--diagnostic_decoders", "--no-diagnostic_decoders")
        )
        if decoder_switches != 1:
            raise ValueError(
                f"step {step_id!r} must explicitly select QD diagnostic decoders"
            )
    if module == "methods.eatr_gmr.evaluate" and "--no-metrics" in argv:
        raise ValueError("EaTR registered evaluation may not use --no-metrics")
    if step.get("kind") != contract["kind"]:
        raise ValueError(
            f"step {step_id!r} kind must be {contract['kind']!r} for module {module}"
        )
    for flag, required_value in contract.get("required_pairs", {}).items():
        if _flag_value(argv, flag) != required_value:
            raise ValueError(f"step {step_id!r} requires {flag} {required_value}")

    raw_static_bindings = step.get("static_input_bindings", {})
    if not isinstance(raw_static_bindings, dict):
        raise ValueError(f"step {step_id!r} static_input_bindings must be an object")
    required_static_flags = contract.get("static_input_flags", {})
    if set(raw_static_bindings) != set(required_static_flags):
        raise ValueError(
            f"step {step_id!r} static input bindings must be exactly "
            f"{sorted(required_static_flags)}"
        )
    normalized_static_bindings: dict[str, list[str]] = {}
    parsed_arguments = semantic_argv["arguments"]
    parser = _parser_for_module(module, repo_root)
    flag_destinations = {
        option: action.dest
        for action in parser._actions
        for option in action.option_strings
    }
    for flag, required_count in required_static_flags.items():
        binding = raw_static_bindings[flag]
        roles = binding if isinstance(binding, list) else [binding]
        if len(roles) != required_count or not all(
            isinstance(role, str) and role in static_input_paths for role in roles
        ) or len(roles) != len(set(roles)):
            raise ValueError(
                f"step {step_id!r} static binding {flag!r} needs "
                f"{required_count} distinct declared roles"
            )
        destination = flag_destinations[flag]
        if destination not in explicit_destinations:
            raise ValueError(f"step {step_id!r} must explicitly pass static flag {flag!r}")
        actual_value = parsed_arguments[destination]
        actual_paths = actual_value if isinstance(actual_value, list) else [actual_value]
        expected_paths = [static_input_paths[role] for role in roles]
        canonical_actual = [
            str(_canonical_non_symlink(
                path, f"step {step_id} static value {flag}", must_exist=True
            ))
            for path in actual_paths
        ]
        if canonical_actual != expected_paths:
            raise ValueError(f"step {step_id!r} static input flag {flag!r} is misbound")
        normalized_static_bindings[flag] = roles

    annotation_flag = step.get("annotation_flag")
    if annotation_flag != contract["annotation_flag"]:
        raise ValueError(
            f"step {step_id!r} annotation_flag must be {contract['annotation_flag']!r}"
        )
    annotation_value = _require_absolute(
        _flag_value(argv, annotation_flag), f"step {step_id} annotation"
    )
    if annotation_value != test_annotations:
        raise ValueError(f"step {step_id!r} is not bound to the frozen test annotation")

    checkpoint_bindings = step.get("checkpoint_bindings", {})
    if not isinstance(checkpoint_bindings, dict):
        raise ValueError(f"step {step_id!r} checkpoint_bindings must be an object")
    if set(checkpoint_bindings) - set(contract["checkpoint_flags"]):
        raise ValueError(f"step {step_id!r} uses an invalid checkpoint flag")
    for flag in contract["checkpoint_flags"]:
        if flag in argv and flag not in checkpoint_bindings:
            raise ValueError(f"step {step_id!r} has unbound checkpoint flag {flag}")
    for flag, role in checkpoint_bindings.items():
        if role not in checkpoint_paths:
            raise ValueError(f"step {step_id!r} references unknown checkpoint role {role!r}")
        actual = _require_absolute(_flag_value(argv, flag), f"step {step_id} {flag}")
        if actual != Path(checkpoint_paths[role]):
            raise ValueError(
                f"step {step_id!r} checkpoint role {role!r} is bound to the wrong path"
            )

    outputs = _normalize_outputs(step, output_root, used_outputs)
    output_bindings = step.get("output_bindings")
    if not isinstance(output_bindings, dict):
        raise ValueError(f"step {step_id!r} output_bindings must be an object")
    if set(output_bindings) - set(contract["output_flags"]):
        raise ValueError(f"step {step_id!r} uses an invalid output flag")
    missing_output_flags = set(contract.get("required_output_flags", set())) - set(
        output_bindings
    )
    if missing_output_flags:
        raise ValueError(f"step {step_id!r} lacks output bindings {sorted(missing_output_flags)}")
    normalized_output_bindings: dict[str, dict[str, str]] = {}
    covered_outputs: set[str] = set()
    for flag, binding in output_bindings.items():
        binding_keys = set(binding) if isinstance(binding, dict) else set()
        if binding_keys not in (
            {"output"}, {"output", "derived_outputs"}, {"directory"},
        ):
            raise ValueError(
                f"step {step_id!r} output binding {flag!r} must name one output or directory"
            )
        actual = _require_absolute(_flag_value(argv, flag), f"step {step_id} {flag}")
        if "output" in binding:
            output_name = binding["output"]
            if output_name not in outputs or actual != Path(outputs[output_name]):
                raise ValueError(f"step {step_id!r} output flag {flag!r} is misbound")
            covered_outputs.add(output_name)
            normalized_binding: dict[str, Any] = {"output": output_name}
            derived_names = binding.get("derived_outputs", [])
            if not isinstance(derived_names, list) or not all(
                isinstance(name, str) and name in outputs and name != output_name
                for name in derived_names
            ) or len(derived_names) != len(set(derived_names)):
                raise ValueError(f"step {step_id!r} has invalid derived_outputs for {flag}")
            if derived_names:
                if module != "methods.qd_detr_gmr.evaluate" or flag not in {
                    "--submission_path", "--metrics_path",
                }:
                    raise ValueError(
                        f"step {step_id!r} derived outputs are unsupported for {module} {flag}"
                    )
                if "--diagnostic_decoders" not in argv or "--no-diagnostic_decoders" in argv:
                    raise ValueError(
                        f"step {step_id!r} derived QD outputs require "
                        "--diagnostic_decoders"
                    )
                allowed_derived = {
                    actual.with_name(f"{actual.stem}.{mode}{actual.suffix}")
                    for mode in ("threshold", "adaptive")
                }
                for derived_name in derived_names:
                    if Path(outputs[derived_name]) not in allowed_derived:
                        raise ValueError(
                            f"step {step_id!r} derived output {derived_name!r} "
                            f"does not follow the QD decoder naming contract"
                        )
                    covered_outputs.add(derived_name)
                normalized_binding["derived_outputs"] = derived_names
            normalized_output_bindings[flag] = normalized_binding
        else:
            directory = _require_absolute(
                binding["directory"], f"step {step_id} output directory"
            )
            if actual != directory or not (
                directory == output_root or output_root in directory.parents
            ):
                raise ValueError(f"step {step_id!r} output directory {flag!r} is misbound")
            for output_name, output_path in outputs.items():
                candidate = Path(output_path)
                if directory in candidate.parents:
                    covered_outputs.add(output_name)
            normalized_output_bindings[flag] = {"directory": str(directory)}
    if covered_outputs != set(outputs):
        raise ValueError(
            f"step {step_id!r} has expected outputs not covered by an output flag: "
            f"{sorted(set(outputs) - covered_outputs)}"
        )

    input_bindings = step.get("input_bindings", {})
    if not isinstance(input_bindings, dict):
        raise ValueError(f"step {step_id!r} input_bindings must be an object")
    if set(input_bindings) - set(contract.get("input_flags", set())):
        raise ValueError(f"step {step_id!r} uses an invalid upstream-input flag")
    missing_input_flags = set(contract.get("required_input_flags", set())) - set(
        input_bindings
    )
    if missing_input_flags:
        raise ValueError(f"step {step_id!r} lacks input bindings {sorted(missing_input_flags)}")
    normalized_inputs = {}
    for flag, binding in input_bindings.items():
        if not isinstance(binding, dict) or set(binding) != {"step", "output"}:
            raise ValueError(f"step {step_id!r} input binding {flag!r} is invalid")
        actual = _require_absolute(_flag_value(argv, flag), f"step {step_id} {flag}")
        normalized_inputs[flag] = {
            "step": binding["step"],
            "output": binding["output"],
            "path": str(actual),
        }

    dependencies = step.get("depends_on", [])
    if not isinstance(dependencies, list) or not all(
        isinstance(value, str) and value for value in dependencies
    ):
        raise ValueError(f"step {step_id!r} depends_on must be a string list")
    if len(dependencies) != len(set(dependencies)):
        raise ValueError(f"step {step_id!r} has duplicate dependencies")

    return {
        "id": step_id,
        "kind": contract["kind"],
        "module": module,
        "argv": argv,
        "canonical_arguments": semantic_argv,
        "python_executable": artifact(executable),
        "working_directory": str(working_directory),
        "environment": _normalize_environment(step),
        "depends_on": dependencies,
        "annotation_flag": annotation_flag,
        "checkpoint_bindings": checkpoint_bindings,
        "static_input_bindings": normalized_static_bindings,
        "input_bindings": normalized_inputs,
        "output_bindings": normalized_output_bindings,
        "outputs": outputs,
        "produces_primary_metrics": bool(contract["produces_primary_metrics"]),
    }


def topological_steps(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {step["id"]: step for step in steps}
    if len(by_id) != len(steps):
        raise ValueError("execution step ids must be unique within an entry")
    for step in steps:
        for dependency in step["depends_on"]:
            if dependency not in by_id:
                raise ValueError(
                    f"step {step['id']!r} references unknown dependency {dependency!r}"
                )
            if dependency == step["id"]:
                raise ValueError(f"step {step['id']!r} may not depend on itself")
    remaining = {step["id"]: set(step["depends_on"]) for step in steps}
    ordered = []
    while remaining:
        ready = [
            step["id"] for step in steps
            if step["id"] in remaining and not remaining[step["id"]]
        ]
        if not ready:
            raise ValueError("execution step dependencies contain a cycle")
        for step_id in ready:
            ordered.append(by_id[step_id])
            del remaining[step_id]
            for dependencies in remaining.values():
                dependencies.discard(step_id)
    return ordered


def _validate_step_graph(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = topological_steps(steps)
    by_id = {step["id"]: step for step in steps}

    def ancestors(step_id: str) -> set[str]:
        result = set()
        pending = list(by_id[step_id]["depends_on"])
        while pending:
            current = pending.pop()
            if current in result:
                continue
            result.add(current)
            pending.extend(by_id[current]["depends_on"])
        return result

    for step in steps:
        available = ancestors(step["id"])
        for flag, binding in step["input_bindings"].items():
            source_id = binding["step"]
            if source_id not in available:
                raise ValueError(
                    f"step {step['id']!r} input {flag!r} must come from a dependency"
                )
            source = by_id[source_id]
            output_name = binding["output"]
            if output_name not in source["outputs"]:
                raise ValueError(
                    f"step {step['id']!r} references unknown output "
                    f"{source_id}.{output_name}"
                )
            if Path(binding["path"]) != Path(source["outputs"][output_name]):
                raise ValueError(
                    f"step {step['id']!r} input {flag!r} is bound to the wrong upstream path"
                )
            required_role = {
                "--localization": "localization",
                "--decision": "decision",
            }.get(flag)
            if required_role is not None and required_role not in set(
                source["checkpoint_bindings"].values()
            ):
                raise ValueError(
                    f"step {step['id']!r} input {flag!r} must originate from "
                    f"checkpoint role {required_role!r}"
                )
    return ordered


def _normalize_entry(
    entry: dict[str, Any], *, test_annotations: Path, working_directory: Path,
    used_outputs: set[Path], validation_annotations: dict[str, Any], repo_root: Path,
) -> dict[str, Any]:
    name = entry.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("every entry needs a non-empty string name")
    role = entry.get("role")
    if role not in ROLE_NAMES:
        raise ValueError(f"invalid role for {name}: {role!r}")
    backbone = entry.get("backbone")
    protocol = entry.get("protocol")
    if backbone not in BACKBONE_ROSTER:
        raise ValueError(
            f"entry {name!r} backbone must be one of {sorted(BACKBONE_ROSTER)}"
        )
    if not isinstance(protocol, str) or not protocol:
        raise ValueError(f"entry {name!r} needs a protocol")
    output_root = Path(ensure_pristine(entry["expected_output_dir"]))

    static_inputs, static_input_paths = _normalize_static_inputs(
        entry, label=f"entry {name!r}"
    )

    raw_checkpoints = entry.get("checkpoint_roles")
    if not isinstance(raw_checkpoints, dict) or not raw_checkpoints:
        raise ValueError(f"entry {name!r} needs checkpoint_roles")
    checkpoint_artifacts = {}
    checkpoint_paths = {}
    for checkpoint_role, path in raw_checkpoints.items():
        if not isinstance(checkpoint_role, str) or not checkpoint_role:
            raise ValueError(f"entry {name!r} has an invalid checkpoint role")
        resolved = _require_absolute(path, f"entry {name} checkpoint {checkpoint_role}")
        frozen = artifact(resolved)
        checkpoint_artifacts[checkpoint_role] = frozen
        checkpoint_paths[checkpoint_role] = frozen["path"]

    raw_steps = entry.get("execution_steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise ValueError(f"entry {name!r} needs execution_steps")
    if not all(isinstance(step, dict) for step in raw_steps):
        raise ValueError(f"entry {name!r} execution_steps must contain objects")
    steps = [
        _normalize_step(
            step, checkpoint_paths=checkpoint_paths,
            test_annotations=test_annotations, output_root=output_root,
            working_directory=working_directory, used_outputs=used_outputs,
            backbone=backbone, static_input_paths=static_input_paths,
            repo_root=repo_root,
        )
        for step in raw_steps
    ]
    ordered_steps = _validate_step_graph(steps)
    bound_roles = {
        checkpoint_role
        for step in steps
        for checkpoint_role in step["checkpoint_bindings"].values()
    }
    if bound_roles != set(checkpoint_paths):
        raise ValueError(
            f"entry {name!r} has unbound checkpoint roles: "
            f"{sorted(set(checkpoint_paths) - bound_roles)}"
        )
    bound_static_roles = {
        role
        for step in steps
        for roles in step["static_input_bindings"].values()
        for role in roles
    }
    if bound_static_roles != set(static_input_paths):
        raise ValueError(
            f"entry {name!r} has unbound static input roles: "
            f"{sorted(set(static_input_paths) - bound_static_roles)}"
        )
    primary_step = entry.get("primary_metric_step")
    primary_matches = [
        step for step in steps
        if step["id"] == primary_step and step["produces_primary_metrics"]
    ]
    if len(primary_matches) != 1:
        raise ValueError(f"entry {name!r} primary_metric_step is not a metric-producing step")
    primary_step_value = primary_matches[0]

    def normalize_primary_output(field: str, suffix: str) -> dict[str, str]:
        binding = entry.get(field)
        if not isinstance(binding, dict) or set(binding) != {"step", "output"}:
            raise ValueError(f"entry {name!r} {field} must bind one step output")
        if binding["step"] != primary_step:
            raise ValueError(f"entry {name!r} {field} must come from primary_metric_step")
        output_name = binding["output"]
        output_path = primary_step_value["outputs"].get(output_name)
        if not isinstance(output_path, str):
            raise ValueError(f"entry {name!r} {field} names an unknown output")
        if not output_path.endswith(suffix):
            raise ValueError(f"entry {name!r} {field} must end in {suffix}")
        return {"step": primary_step, "output": output_name, "path": output_path}

    primary_metric_output = normalize_primary_output(
        "primary_metric_output", ".json"
    )
    primary_submission_output = normalize_primary_output(
        "primary_submission_output", ".jsonl"
    )
    if primary_metric_output["path"] == primary_submission_output["path"]:
        raise ValueError(f"entry {name!r} primary outputs must be distinct")
    metric_steps = [step["id"] for step in steps if step["produces_primary_metrics"]]
    if {"localization", "decision"}.issubset(checkpoint_paths):
        composition_steps = [
            step for step in steps if step["module"] == "scripts.fuse_gmr_heads"
        ]
        if len(composition_steps) != 1 or primary_step != composition_steps[0]["id"]:
            raise ValueError(
                f"entry {name!r} localization/decision composition must use its "
                "single fusion step as primary_metric_step"
            )

    metrics_path = _require_absolute(entry["validation_metrics"], f"entry {name} metrics")
    metrics = brief(metrics_path)
    _validate_metric_brief(metrics, f"entry {name}")
    validation_metrics = artifact(metrics_path)
    provenance_path = _require_absolute(
        entry.get("checkpoint_validation_provenance", ""),
        f"entry {name} checkpoint-validation provenance",
    )
    provenance = load_json(provenance_path)
    normalized_provenance = validate_checkpoint_validation_provenance(
        provenance,
        checkpoint_roles=checkpoint_artifacts,
        validation_metrics=validation_metrics,
        validation_annotations=validation_annotations,
        label=f"entry {name}",
    )
    normalized = {
        "name": name,
        "role": role,
        "backbone": backbone,
        "protocol": protocol,
        "checkpoint_roles": checkpoint_artifacts,
        "static_inputs": static_inputs,
        "validation_metrics": validation_metrics,
        "validation_brief": metrics,
        "checkpoint_validation_provenance": artifact(provenance_path),
        "checkpoint_validation_provenance_summary": normalized_provenance,
        "primary_metric_step": primary_step,
        "primary_metric_output": primary_metric_output,
        "primary_submission_output": primary_submission_output,
        "metric_producing_steps": metric_steps,
        "execution_steps": ordered_steps,
        "expected_output_dir": str(output_root),
        "execution_count_before_freeze": 0,
    }
    if role == "candidate":
        reference_entry = entry.get("reference_entry")
        diagnostics_path = entry.get("group_diagnostics")
        if not isinstance(reference_entry, str) or not reference_entry or not diagnostics_path:
            raise ValueError(
                f"candidate {name!r} needs reference_entry and group_diagnostics"
            )
        normalized["reference_entry"] = reference_entry
        diagnostics_path = _require_absolute(
            diagnostics_path, f"entry {name} group diagnostics"
        )
        diagnostics = load_json(diagnostics_path)
        _diagnostic_values(diagnostics, f"entry {name}")
        normalized["group_diagnostics"] = artifact(diagnostics_path)
        normalized["group_diagnostics_summary"] = diagnostics
        three_seed_path = _require_absolute(
            entry.get("three_seed_validation", ""),
            f"entry {name} three-seed validation",
        )
        three_seed_raw = load_json(three_seed_path)
        three_seed_summary, three_seed_gates = validate_three_seed_summary(
            three_seed_raw, label=f"entry {name}",
        )
        if not all(three_seed_gates.values()):
            raise RuntimeError(
                f"three-seed validation gates failed for {name}: {three_seed_gates}"
            )
        normalized["three_seed_validation"] = artifact(three_seed_path)
        normalized["three_seed_validation_summary"] = three_seed_summary
        normalized["three_seed_validation_gates"] = three_seed_gates
    elif "reference_entry" in entry:
        raise ValueError(f"only candidate entries may declare reference_entry: {name}")
    elif entry.get("group_diagnostics"):
        diagnostics_path = _require_absolute(
            entry["group_diagnostics"], f"entry {name} group diagnostics"
        )
        diagnostics = load_json(diagnostics_path)
        _diagnostic_values(diagnostics, f"entry {name}")
        normalized["group_diagnostics"] = artifact(diagnostics_path)
        normalized["group_diagnostics_summary"] = diagnostics
    return normalized


def _seal_value(manifest: dict[str, Any]) -> str:
    unsealed = copy.deepcopy(manifest)
    unsealed.pop("seal", None)
    return sha256_bytes(_canonical_json(unsealed))


def verify_manifest_seal(manifest: dict[str, Any]) -> None:
    expected = manifest.get("seal")
    if not isinstance(expected, str) or expected != _seal_value(manifest):
        raise RuntimeError("manifest seal is missing or invalid")


def _manifest_payload(manifest: dict[str, Any]) -> bytes:
    return (
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def protocol_registration_record(
    manifest: dict[str, Any], manifest_path: str | Path,
) -> dict[str, Any]:
    canonical_manifest_path = _canonical_non_symlink(
        manifest_path, "manifest output", must_exist=False
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": manifest["protocol_id"],
        "manifest_path": str(canonical_manifest_path),
        "manifest_sha256": sha256_bytes(_manifest_payload(manifest)),
        "manifest_seal": manifest["seal"],
    }


def verify_protocol_registration(
    manifest: dict[str, Any], manifest_path: str | Path,
    *, trusted_registry_root: str | Path | None = None,
) -> dict[str, Any]:
    registry_root = _canonical_non_symlink(
        trusted_registry_root or DEFAULT_TRUSTED_REGISTRY_ROOT,
        "trusted registry root", must_exist=True,
    )
    if manifest.get("registry_root") != str(registry_root):
        raise RuntimeError("manifest registry_root is not the trusted registry root")
    expected_registration_path = registry_root / (
        f"{manifest.get('protocol_id')}.registration.json"
    )
    if manifest.get("registration_path") != str(expected_registration_path):
        raise RuntimeError("manifest registration_path is not registry-derived")
    expected_ledger_path = registry_root / f"{manifest.get('protocol_id')}.ledger.jsonl"
    if manifest.get("ledger_path") != str(expected_ledger_path):
        raise RuntimeError("manifest ledger_path is not registry-derived")
    registration_path = _canonical_non_symlink(
        expected_registration_path, "protocol registration", must_exist=True
    )
    if not registration_path.is_file():
        raise RuntimeError("protocol registration is not a regular file")
    record = load_json(registration_path)
    expected = protocol_registration_record(manifest, manifest_path)
    if record != expected:
        raise RuntimeError(
            f"trusted protocol registration differs: expected={expected} actual={record}"
        )
    if sha256_file(manifest_path) != record["manifest_sha256"]:
        raise RuntimeError("manifest bytes differ from trusted protocol registration")
    return record


def build_manifest(
    spec_path: str | Path,
    test_annotations: str | Path,
    sources: list[str],
    *,
    trusted_registry_root: str | Path | None = None,
    _created_utc: str | None = None,
    _allow_registered: bool = False,
) -> dict[str, Any]:
    if not isinstance(sources, list) or not sources or not all(
        isinstance(path, str) and path for path in sources
    ):
        raise ValueError("at least one source path is required")
    spec = load_json(spec_path)
    if spec.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"spec schema_version must be {SCHEMA_VERSION}")
    if spec.get("selection_split") != "validation":
        raise ValueError("spec selection_split must be exactly 'validation'")
    protocol_id = spec.get("protocol_id")
    if not isinstance(protocol_id, str) or not PROTOCOL_ID_RE.fullmatch(protocol_id):
        raise ValueError("spec protocol_id is invalid")
    if spec.get("max_executions") != 1:
        raise ValueError("spec max_executions must be exactly 1")
    if "ledger_dir" in spec or "registry_root" in spec:
        raise ValueError("spec may not choose the trusted registry/ledger root")
    working_directory = _require_absolute(
        spec.get("working_directory", ""), "working_directory"
    )
    if working_directory != REPO_ROOT:
        raise ValueError(
            f"working_directory must be the canonical repository root: {REPO_ROOT}"
        )
    registry_root = _canonical_non_symlink(
        trusted_registry_root or DEFAULT_TRUSTED_REGISTRY_ROOT,
        "trusted registry root",
        must_exist=False,
    )
    if registry_root.exists() and not registry_root.is_dir():
        raise ValueError(f"trusted registry root is not a directory: {registry_root}")
    registration_path = registry_root / f"{protocol_id}.registration.json"
    ledger_path = registry_root / f"{protocol_id}.ledger.jsonl"
    if not _allow_registered and (registration_path.exists() or ledger_path.exists()):
        raise RuntimeError(
            f"one-shot protocol_id is already registered or claimed: {protocol_id}"
        )

    resolved_test_annotations = _require_absolute(test_annotations, "test_annotations")
    spec_test_annotations = _require_absolute(
        spec.get("test_annotations", ""), "spec test_annotations"
    )
    if resolved_test_annotations != spec_test_annotations:
        raise ValueError("CLI and spec test annotations differ")
    validation_annotations = _require_absolute(
        spec.get("validation_annotations", ""), "validation_annotations"
    )
    if validation_annotations == resolved_test_annotations:
        raise ValueError("validation and test annotations must be different files")

    entries = spec.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("spec must contain a non-empty entries list")
    if not all(isinstance(entry, dict) for entry in entries):
        raise ValueError("every spec entry must be an object")
    names = [entry.get("name") for entry in entries]
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("every entry needs a non-empty string name")
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate entry names: {names}")

    frozen_validation_annotations = artifact(validation_annotations)
    used_outputs: set[Path] = set()
    frozen_entries = [
        _normalize_entry(
            entry, test_annotations=resolved_test_annotations,
            working_directory=working_directory, used_outputs=used_outputs,
            validation_annotations=frozen_validation_annotations,
            repo_root=REPO_ROOT,
        )
        for entry in entries
    ]
    output_roots = [Path(entry["expected_output_dir"]) for entry in frozen_entries]
    for index, first in enumerate(output_roots):
        if paths_overlap(first, registry_root):
            raise ValueError(
                f"output root overlaps trusted registry root: {first} / {registry_root}"
            )
        for second in output_roots[index + 1:]:
            if paths_overlap(first, second):
                raise ValueError(f"entry output roots overlap: {first} / {second}")

    by_name = {entry["name"]: entry for entry in frozen_entries}
    for entry in frozen_entries:
        if entry["role"] != "candidate":
            continue
        reference_name = entry["reference_entry"]
        reference = by_name.get(reference_name)
        if reference is None or reference["role"] != "anchor":
            raise ValueError(
                f"candidate {entry['name']!r} reference_entry must name an anchor"
            )
        if reference["backbone"] != entry["backbone"]:
            raise ValueError(f"candidate {entry['name']!r} reference backbone differs")
        if reference["protocol"] != entry["protocol"]:
            raise ValueError(f"candidate {entry['name']!r} reference protocol differs")
        gates = candidate_gates(
            entry["validation_brief"], reference["validation_brief"],
            entry["group_diagnostics_summary"], label=f"entry {entry['name']}",
        )
        if not all(gates.values()):
            raise RuntimeError(f"validation gates failed for {entry['name']}: {gates}")
        entry["reference_validation_metrics"] = reference["validation_metrics"]
        entry["reference_validation_brief"] = reference["validation_brief"]
        entry["validation_gates"] = gates

    frozen_sources = source_artifacts(sources)
    frozen_source_roots = [source_inventory(path) for path in sources]
    for source_root in frozen_source_roots:
        if paths_overlap(registry_root, Path(source_root["path"])):
            raise ValueError(
                f"trusted registry root overlaps a frozen source root: "
                f"{registry_root} / {source_root['path']}"
            )
    used_modules = sorted({
        step["module"]
        for entry in frozen_entries
        for step in entry["execution_steps"]
    } | {
        "scripts.preregister_detr_matrix_test",
        "scripts.run_preregistered_detr_matrix_test",
    })
    module_source_bindings = {
        module: artifact(_module_source(module, REPO_ROOT))
        for module in used_modules
    }
    frozen_inputs = [
        artifact(spec_path), frozen_validation_annotations,
        artifact(resolved_test_annotations), *frozen_sources,
        *module_source_bindings.values(),
    ]
    frozen_inputs.extend(
        checkpoint
        for entry in frozen_entries
        for checkpoint in entry["checkpoint_roles"].values()
    )
    frozen_inputs.extend(
        step["python_executable"]
        for entry in frozen_entries
        for step in entry["execution_steps"]
    )
    frozen_inputs.extend(entry["validation_metrics"] for entry in frozen_entries)
    frozen_inputs.extend(
        entry["checkpoint_validation_provenance"] for entry in frozen_entries
    )
    frozen_inputs.extend(
        inventory
        for entry in frozen_entries
        for inventory in entry["static_inputs"].values()
        if inventory["kind"] == "file"
    )
    frozen_inputs.extend(
        entry["group_diagnostics"]
        for entry in frozen_entries if "group_diagnostics" in entry
    )
    frozen_inputs.extend(
        entry["three_seed_validation"]
        for entry in frozen_entries if "three_seed_validation" in entry
    )
    for root in output_roots:
        for source_root in frozen_source_roots:
            if paths_overlap(root, Path(source_root["path"])):
                raise ValueError(
                    f"output root overlaps a frozen source root: "
                    f"{root} / {source_root['path']}"
                )
        for frozen in frozen_inputs:
            if paths_overlap(root, Path(frozen["path"])):
                raise ValueError(
                    f"output root overlaps a frozen input artifact: {root} / {frozen['path']}"
                )
        for entry in frozen_entries:
            for role, inventory in entry["static_inputs"].items():
                if paths_overlap(root, Path(inventory["path"])):
                    raise ValueError(
                        f"output root overlaps static input {entry['name']}.{role}: "
                        f"{root} / {inventory['path']}"
                    )

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "frozen_before_test",
        "created_utc": _created_utc or dt.datetime.now(dt.timezone.utc).isoformat(),
        "protocol_id": protocol_id,
        "max_executions": 1,
        "selection_split": "validation",
        "working_directory": str(working_directory),
        "registry_root": str(registry_root),
        "registration_path": str(registration_path),
        "ledger_path": str(ledger_path),
        "spec": artifact(spec_path),
        "validation_annotations": frozen_validation_annotations,
        # Bytewise hashing only: no test label-dependent computation occurs.
        "test_annotations": artifact(resolved_test_annotations),
        "entries": frozen_entries,
        "source_files": frozen_sources,
        "source_roots": frozen_source_roots,
        "module_source_bindings": module_source_bindings,
    }
    manifest["seal"] = _seal_value(manifest)
    return manifest


def validate_frozen_step_semantics(
    step: dict[str, Any], *, repo_root: Path = REPO_ROOT,
) -> None:
    argv = step.get("argv")
    module = step.get("module")
    if not isinstance(argv, list) or len(argv) < 3 or argv[1:3] != ["-m", module]:
        raise RuntimeError("frozen step argv/module shape is invalid")
    executable = artifact(argv[0])
    if executable != step.get("python_executable"):
        raise RuntimeError("frozen Python executable artifact differs")
    if step.get("working_directory") != str(repo_root):
        raise RuntimeError("frozen step working_directory differs from repository root")
    environment = step.get("environment")
    if not isinstance(environment, dict) or "PYTHONPATH" in environment or any(
        key not in ALLOWED_ENV_KEYS for key in environment
    ):
        raise RuntimeError("frozen step environment is unsafe")
    rebuilt = canonicalize_argv(argv, module=module, repo_root=repo_root)
    if rebuilt != step.get("canonical_arguments"):
        raise RuntimeError("frozen argv canonical semantics differ")


def rebuild_manifest_from_frozen(
    manifest: dict[str, Any], *, trusted_registry_root: str | Path | None = None,
) -> dict[str, Any]:
    try:
        sources = [value["path"] for value in manifest["source_roots"]]
        rebuilt = build_manifest(
            manifest["spec"]["path"],
            manifest["test_annotations"]["path"],
            sources,
            trusted_registry_root=trusted_registry_root,
            _created_utc=manifest["created_utc"],
            _allow_registered=True,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("frozen manifest cannot be rebuilt from its spec") from error
    if _canonical_json(rebuilt) != _canonical_json(manifest):
        raise RuntimeError(
            "frozen manifest differs semantically from a fresh normalized spec rebuild"
        )
    return rebuilt


def write_manifest_exclusive(manifest: dict[str, Any], output: str | Path) -> None:
    verify_manifest_seal(manifest)
    path = _canonical_non_symlink(output, "manifest output", must_exist=False)
    if path.exists():
        raise FileExistsError(path)
    registry_root = _canonical_non_symlink(
        manifest.get("registry_root", ""), "manifest registry root", must_exist=False
    )
    if manifest.get("registration_path") != str(
        registry_root / f"{manifest.get('protocol_id')}.registration.json"
    ):
        raise RuntimeError("manifest registration path is not protocol-derived")
    registration_path = Path(manifest["registration_path"])
    registry_root.mkdir(parents=True, exist_ok=True)
    _canonical_non_symlink(registry_root, "manifest registry root", must_exist=True)
    record = protocol_registration_record(manifest, path)
    registration_payload = (
        json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    )
    try:
        descriptor = os.open(
            registration_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
        )
    except FileExistsError as error:
        raise FileExistsError(
            f"protocol_id is already registered: {manifest['protocol_id']}"
        ) from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as registration_handle:
        registration_handle.write(registration_payload)
        registration_handle.flush()
        os.fsync(registration_handle.fileno())
    path.parent.mkdir(parents=True, exist_ok=True)
    _canonical_non_symlink(path.parent, "manifest output parent", must_exist=True)
    payload = _manifest_payload(manifest).decode("utf-8")
    # x/O_EXCL makes a frozen manifest write-once.
    with path.open("x", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--test-annotations", required=True)
    parser.add_argument("--source", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    manifest = build_manifest(args.spec, args.test_annotations, args.source)
    write_manifest_exclusive(manifest, args.output)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
