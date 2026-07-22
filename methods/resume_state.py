"""Exact checkpoint-selection state shared by isolated DETR trainers."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


SELECTION_METRICS = ("mAP", "G-mIoU@3", "joint")
TRAINING_STATE_VERSION = 1


def make_training_state(
    best: dict[str, float], epochs_without_improvement: int
) -> dict[str, Any]:
    """Serialize checkpoint-selection/early-stop state into a checkpoint."""

    normalized = {}
    for name in SELECTION_METRICS:
        if name not in best:
            raise ValueError(f"selection state is missing {name!r}")
        value = float(best[name])
        if math.isnan(value) or value == math.inf:
            raise ValueError(f"selection state {name!r} is invalid: {value}")
        normalized[name] = value
    counter = int(epochs_without_improvement)
    if counter < 0:
        raise ValueError("epochs_without_improvement must be non-negative")
    return {
        "version": TRAINING_STATE_VERSION,
        "best": normalized,
        "epochs_without_improvement": counter,
    }


def _validate_training_state(value: Any) -> tuple[dict[str, float], int]:
    if not isinstance(value, dict) or value.get("version") != TRAINING_STATE_VERSION:
        raise ValueError("checkpoint training_state has an unsupported version")
    best = value.get("best")
    if not isinstance(best, dict):
        raise ValueError("checkpoint training_state.best must be an object")
    normalized = {}
    for name in SELECTION_METRICS:
        try:
            score = float(best[name])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"checkpoint training_state.best lacks numeric {name!r}"
            ) from error
        if math.isnan(score) or score == math.inf:
            raise ValueError(f"checkpoint training_state.best[{name!r}] is invalid")
        normalized[name] = score
    counter = value.get("epochs_without_improvement")
    if not isinstance(counter, int) or isinstance(counter, bool) or counter < 0:
        raise ValueError(
            "checkpoint training_state.epochs_without_improvement must be a "
            "non-negative integer"
        )
    return normalized, counter


def restore_training_state(
    checkpoint: dict[str, Any],
    log_path: str | Path,
    *,
    primary_metric: str,
) -> tuple[dict[str, float], int]:
    """Restore saved state, or reconstruct legacy state from the JSONL log.

    Older checkpoints saved model/optimizer/scheduler but omitted the best
    scores and early-stop counter.  Replaying the already-written selection
    records is deterministic and prevents an interrupted resume from resetting
    patience or overwriting a genuinely better checkpoint.
    """

    if primary_metric not in SELECTION_METRICS:
        raise ValueError(f"unknown primary selection metric: {primary_metric!r}")
    if "training_state" in checkpoint:
        return _validate_training_state(checkpoint["training_state"])

    path = Path(log_path)
    if not path.is_file():
        raise RuntimeError(
            "legacy resume checkpoint has no training_state and its JSONL log "
            f"is unavailable: {path}"
        )
    checkpoint_epoch = checkpoint.get("epoch")
    if not isinstance(checkpoint_epoch, int) or isinstance(checkpoint_epoch, bool):
        raise RuntimeError("legacy resume checkpoint has no integer epoch")
    maximum_record_epoch = checkpoint_epoch + 1
    selections: list[tuple[int, dict[str, float]]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"cannot reconstruct resume state from malformed {path}:{line_number}"
                ) from error
            epoch = record.get("epoch")
            raw_selection = record.get("selection")
            if not isinstance(epoch, int) or epoch > maximum_record_epoch:
                continue
            if not isinstance(raw_selection, dict):
                continue
            selection = {}
            for name in SELECTION_METRICS:
                try:
                    score = float(raw_selection[name])
                except (KeyError, TypeError, ValueError) as error:
                    raise RuntimeError(
                        f"cannot reconstruct {name!r} from {path}:{line_number}"
                    ) from error
                if not math.isfinite(score):
                    raise RuntimeError(
                        f"non-finite {name!r} in {path}:{line_number}"
                    )
                selection[name] = score
            selections.append((epoch, selection))
    if not selections:
        raise RuntimeError(
            "legacy resume checkpoint has no training_state and no selection "
            f"records through epoch {maximum_record_epoch}: {path}"
        )
    if selections[-1][0] > maximum_record_epoch:
        raise AssertionError("selection reconstruction crossed checkpoint epoch")

    best = {name: float("-inf") for name in SELECTION_METRICS}
    epochs_without_improvement = 0
    for _, selection in selections:
        improved_primary = selection[primary_metric] > best[primary_metric]
        for name in SELECTION_METRICS:
            best[name] = max(best[name], selection[name])
        epochs_without_improvement = (
            0 if improved_primary else epochs_without_improvement + 1
        )
    return best, epochs_without_improvement
