#!/usr/bin/env python3
"""Execute a frozen DETR matrix exactly once, without a shell.

All schema and pristine-output checks happen before an atomic matrix-level
claim is created.  Once claimed, success, command failure, Python failure, or
process death all consume the single registered execution.  The ledger is an
append-only hash chain stored outside every test output root.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.preregister_detr_matrix_test import (
    ALLOWED_ENV_KEYS,
    DEFAULT_TRUSTED_REGISTRY_ROOT,
    REPO_ROOT,
    SCHEMA_VERSION,
    _canonical_non_symlink,
    _validate_metric_brief,
    artifact,
    load_json,
    paths_overlap,
    rebuild_manifest_from_frozen,
    sha256_file,
    topological_steps,
    verify_artifact,
    verify_manifest_seal,
    verify_protocol_registration,
    verify_source_inventory,
    validate_frozen_step_semantics,
)


class AlreadyExecutedError(RuntimeError):
    """Raised when the matrix-level one-shot claim already exists."""


class RegisteredCommandError(RuntimeError):
    """Raised when an exact registered command exits unsuccessfully."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _stream_summary(value: bytes | str | None) -> dict[str, Any]:
    if value is None:
        payload = b""
    elif isinstance(value, bytes):
        payload = value
    else:
        payload = value.encode("utf-8", errors="replace")
    return {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def _iter_static_artifacts(manifest: dict[str, Any]):
    yield manifest["spec"]
    yield manifest["validation_annotations"]
    yield manifest["test_annotations"]
    yield from manifest.get("source_files", [])
    yield from manifest.get("module_source_bindings", {}).values()
    for entry in manifest["entries"]:
        yield entry["validation_metrics"]
        yield entry["checkpoint_validation_provenance"]
        yield from entry["checkpoint_roles"].values()
        for step in entry["execution_steps"]:
            yield step["python_executable"]
        if "group_diagnostics" in entry:
            yield entry["group_diagnostics"]
        if "three_seed_validation" in entry:
            yield entry["three_seed_validation"]


def verify_all_static_artifacts(manifest: dict[str, Any]) -> None:
    seen = set()
    for frozen in _iter_static_artifacts(manifest):
        key = (frozen["path"], frozen["sha256"])
        if key in seen:
            continue
        seen.add(key)
        verify_artifact(frozen)
    for source_root in manifest.get("source_roots", []):
        verify_source_inventory(source_root)
    for entry in manifest["entries"]:
        for static_input in entry["static_inputs"].values():
            verify_source_inventory(static_input)


def _output_roots(manifest: dict[str, Any]) -> list[Path]:
    return [
        _canonical_non_symlink(
            entry["expected_output_dir"], "registered output root", must_exist=False
        )
        for entry in manifest["entries"]
    ]


def _expected_paths(entry: dict[str, Any]) -> set[Path]:
    return {
        _canonical_non_symlink(path, "registered output", must_exist=False)
        for step in entry["execution_steps"]
        for path in step["outputs"].values()
    }


def _assert_root_pristine(root: Path) -> None:
    _canonical_non_symlink(root, "registered output root", must_exist=False)
    if root.exists() and not root.is_dir():
        raise RuntimeError(f"registered output root is not a directory: {root}")
    children = list(root.iterdir()) if root.exists() else []
    if children:
        raise RuntimeError(
            f"registered output root is no longer pristine: "
            f"{[str(child) for child in children[:5]]}"
        )


def verify_pristine_roots(manifest: dict[str, Any]) -> None:
    roots = _output_roots(manifest)
    ledger_path = _canonical_non_symlink(
        manifest["ledger_path"], "registered ledger", must_exist=False
    )
    for index, first in enumerate(roots):
        _assert_root_pristine(first)
        if paths_overlap(first, ledger_path.parent):
            raise RuntimeError(f"output root overlaps ledger directory: {first}")
        for second in roots[index + 1:]:
            if paths_overlap(first, second):
                raise RuntimeError(f"registered output roots overlap: {first} / {second}")


def _verify_output_inventory(entry: dict[str, Any], *, require_all: bool) -> None:
    root = Path(entry["expected_output_dir"])
    expected = _expected_paths(entry)
    if not root.exists():
        if require_all:
            raise RuntimeError(f"registered output root was not created: {root}")
        return
    for child in root.rglob("*"):
        if child.is_symlink():
            raise RuntimeError(f"registered command created a symlink: {child}")
        if child.is_file() and child.resolve() not in expected:
            raise RuntimeError(f"registered command created an undeclared file: {child}")
        if child.is_dir() and not any(
            child.resolve() == path.parent or child.resolve() in path.parents
            for path in expected
        ):
            raise RuntimeError(f"registered command created an undeclared directory: {child}")
    if require_all:
        missing = [path for path in sorted(expected) if not path.is_file() or path.is_symlink()]
        if missing:
            raise RuntimeError(f"registered outputs are missing: {missing}")


def _event_hash(event: dict[str, Any]) -> str:
    value = dict(event)
    value.pop("event_hash", None)
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def read_and_verify_ledger(path: str | Path) -> list[dict[str, Any]]:
    ledger = Path(path)
    events = []
    previous = "0" * 64
    with ledger.open("r", encoding="utf-8") as handle:
        for expected_seq, line in enumerate(handle):
            if not line.endswith("\n"):
                raise RuntimeError(f"ledger has a truncated final record: {ledger}")
            try:
                event = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(f"ledger is not valid JSONL: {ledger}") from error
            if event.get("seq") != expected_seq or event.get("prev_hash") != previous:
                raise RuntimeError(f"ledger sequence/hash chain is invalid: {ledger}")
            if event.get("event_hash") != _event_hash(event):
                raise RuntimeError(f"ledger event hash is invalid: {ledger}")
            previous = event["event_hash"]
            events.append(event)
    if not events:
        raise RuntimeError(f"one-shot ledger exists but is empty: {ledger}")
    return events


class _LedgerWriter:
    def __init__(self, path: Path, *, protocol_id: str, manifest_sha256: str):
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(
                path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_APPEND, 0o600
            )
        except FileExistsError as error:
            # A corrupt claim is still a consumed claim; verify it only to make
            # tampering visible, never to permit a retry.
            read_and_verify_ledger(path)
            raise AlreadyExecutedError(f"one-shot matrix already claimed: {path}") from error
        self.path = path
        self.handle = os.fdopen(descriptor, "w", encoding="utf-8")
        self.protocol_id = protocol_id
        self.manifest_sha256 = manifest_sha256
        self.seq = 0
        self.previous = "0" * 64

    def append(self, event_type: str, **values: Any) -> dict[str, Any]:
        event = {
            "seq": self.seq,
            "prev_hash": self.previous,
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "protocol_id": self.protocol_id,
            "manifest_sha256": self.manifest_sha256,
            "event": event_type,
            **values,
        }
        event["event_hash"] = _event_hash(event)
        self.handle.write(json.dumps(
            event, ensure_ascii=False, sort_keys=True, allow_nan=False
        ) + "\n")
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.seq += 1
        self.previous = event["event_hash"]
        return event

    def close(self) -> None:
        if not self.handle.closed:
            self.handle.close()


def _verify_completed_artifact(frozen: dict[str, Any]) -> None:
    verify_artifact(frozen)


def _assert_manifest_runtime_shape(manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError(f"manifest schema_version must be {SCHEMA_VERSION}")
    if manifest.get("status") != "frozen_before_test":
        raise RuntimeError("manifest is not frozen_before_test")
    if manifest.get("max_executions") != 1:
        raise RuntimeError("registered matrix max_executions must be exactly 1")
    if not isinstance(manifest.get("entries"), list) or not manifest["entries"]:
        raise RuntimeError("registered matrix has no entries")
    working_directory = Path(manifest["working_directory"])
    if not working_directory.is_absolute() or not working_directory.is_dir():
        raise RuntimeError("registered working_directory is invalid")
    for entry in manifest["entries"]:
        ordered = topological_steps(entry["execution_steps"])
        if [step["id"] for step in ordered] != [
            step["id"] for step in entry["execution_steps"]
        ]:
            raise RuntimeError(f"entry {entry['name']} steps are not stored topologically")


def run_manifest(
    manifest_path: str | Path,
    *,
    subprocess_runner: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    path = Path(manifest_path)
    if not path.is_absolute():
        path = path.resolve()
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"manifest must be a regular non-symlink file: {path}")
    manifest = load_json(path)
    verify_manifest_seal(manifest)
    _assert_manifest_runtime_shape(manifest)
    verify_all_static_artifacts(manifest)
    ledger_path = Path(manifest["ledger_path"])
    if ledger_path.exists():
        read_and_verify_ledger(ledger_path)
        raise AlreadyExecutedError(f"one-shot matrix already claimed: {ledger_path}")
    verify_pristine_roots(manifest)
    runner = subprocess.run if subprocess_runner is None else subprocess_runner
    manifest_sha256 = sha256_file(path)
    ledger = _LedgerWriter(
        ledger_path,
        protocol_id=manifest["protocol_id"],
        manifest_sha256=manifest_sha256,
    )
    completed_outputs: dict[tuple[str, str, str], dict[str, Any]] = {}
    try:
        ledger.append("matrix_started", max_executions=1)
        # Close the freeze/claim race before the first subprocess.
        verify_manifest_seal(manifest)
        verify_all_static_artifacts(manifest)
        verify_pristine_roots(manifest)

        for entry in manifest["entries"]:
            entry_name = entry["name"]
            ledger.append("entry_started", entry=entry_name)
            for step in entry["execution_steps"]:
                step_id = step["id"]
                if sha256_file(path) != manifest_sha256:
                    raise RuntimeError("frozen manifest changed during registered execution")
                verify_all_static_artifacts(manifest)
                for binding in step["input_bindings"].values():
                    key = (entry_name, binding["step"], binding["output"])
                    frozen_output = completed_outputs.get(key)
                    if frozen_output is None:
                        raise RuntimeError(
                            f"step {entry_name}.{step_id} dependency output is unavailable: {key}"
                        )
                    _verify_completed_artifact(frozen_output)
                for output_path in step["outputs"].values():
                    if Path(output_path).exists() or Path(output_path).is_symlink():
                        raise RuntimeError(
                            f"step output exists before registered execution: {output_path}"
                        )

                ledger.append(
                    "step_started", entry=entry_name, step=step_id,
                    module=step["module"], argv=step["argv"],
                )
                environment = os.environ.copy()
                for name in ALLOWED_ENV_KEYS:
                    environment.pop(name, None)
                environment.update(step["environment"])
                try:
                    result = runner(
                        step["argv"],
                        cwd=step["working_directory"],
                        env=environment,
                        shell=False,
                        capture_output=True,
                        check=False,
                    )
                except BaseException as error:
                    ledger.append(
                        "step_failed", entry=entry_name, step=step_id,
                        error_type=type(error).__name__, error=str(error),
                    )
                    raise
                returncode = int(result.returncode)
                stdout = _stream_summary(getattr(result, "stdout", None))
                stderr = _stream_summary(getattr(result, "stderr", None))
                if returncode != 0:
                    ledger.append(
                        "step_failed", entry=entry_name, step=step_id,
                        returncode=returncode, stdout=stdout, stderr=stderr,
                    )
                    raise RegisteredCommandError(
                        f"registered command failed: {entry_name}.{step_id} exit={returncode}"
                    )

                try:
                    verify_all_static_artifacts(manifest)
                    frozen_step_outputs = {}
                    for output_name, output_path in step["outputs"].items():
                        frozen = artifact(output_path)
                        frozen_step_outputs[output_name] = frozen
                        completed_outputs[(entry_name, step_id, output_name)] = frozen
                    _verify_output_inventory(entry, require_all=False)
                except BaseException as error:
                    ledger.append(
                        "step_failed", entry=entry_name, step=step_id,
                        returncode=returncode, stdout=stdout, stderr=stderr,
                        error_type=type(error).__name__, error=str(error),
                    )
                    raise
                ledger.append(
                    "step_completed", entry=entry_name, step=step_id,
                    returncode=returncode, stdout=stdout, stderr=stderr,
                    outputs=frozen_step_outputs,
                )

            _verify_output_inventory(entry, require_all=True)
            for key, frozen in completed_outputs.items():
                if key[0] == entry_name:
                    _verify_completed_artifact(frozen)
            ledger.append("entry_completed", entry=entry_name)

        all_outputs = {
            f"{entry}.{step}.{name}": frozen
            for (entry, step, name), frozen in completed_outputs.items()
        }
        if sha256_file(path) != manifest_sha256:
            raise RuntimeError("frozen manifest changed during registered execution")
        verify_all_static_artifacts(manifest)
        ledger.append("matrix_completed", outputs=all_outputs)
        return {
            "status": "completed",
            "protocol_id": manifest["protocol_id"],
            "manifest_sha256": manifest_sha256,
            "ledger_path": str(ledger_path),
            "outputs": all_outputs,
        }
    except BaseException as error:
        try:
            ledger.append(
                "matrix_failed", error_type=type(error).__name__, error=str(error)
            )
        except BaseException:
            pass
        raise
    finally:
        ledger.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    result = run_manifest(args.manifest)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
