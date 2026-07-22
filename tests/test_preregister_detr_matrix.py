from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.preregister_detr_matrix_test import (
    build_manifest,
    verify_manifest_seal,
    write_manifest_exclusive,
)
from scripts.run_preregistered_detr_matrix_test import (
    AlreadyExecutedError,
    RegisteredCommandError,
    read_and_verify_ledger,
    run_manifest,
)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class PreregisterDETRMatrixTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.validation = self.root / "validation.jsonl"
        self.validation.write_text('{"qid":"validation"}\n', encoding="utf-8")
        self.test_annotation = self.root / "heldout.jsonl"
        self.test_annotation.write_text('{"qid":"heldout"}\n', encoding="utf-8")
        self.source = self.root / "source.py"
        self.source.write_text("VALUE = 1\n", encoding="utf-8")
        self.anchor_checkpoint = self.root / "anchor.ckpt"
        self.anchor_checkpoint.write_bytes(b"anchor-checkpoint")
        self.candidate_checkpoint = self.root / "candidate.ckpt"
        self.candidate_checkpoint.write_bytes(b"candidate-checkpoint")
        self.anchor_metrics = self.root / "anchor.metrics.json"
        write_json(self.anchor_metrics, {
            "brief": {"mAP": 8.0, "G-mIoU@3": 30.0, "mR+@5": 1.0}
        })
        self.candidate_metrics = self.root / "candidate.metrics.json"
        write_json(self.candidate_metrics, {
            "brief": {"mAP": 9.0, "G-mIoU@3": 31.0, "mR+@5": 0.8}
        })
        self.diagnostics = self.root / "diagnostics.json"
        write_json(self.diagnostics, {
            "balanced_G": 51.0,
            "groups": {
                "single": {"support": 10, "acceptance_rate": 20.0},
                "multi": {"support": 5, "acceptance_rate": 10.0},
            },
        })
        self.spec_path = self.root / "spec.json"
        self.manifest_path = self.root / "manifest.json"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def qd_entry(
        self, name: str, role: str, checkpoint: Path, metrics: Path,
        *, backbone: str = "qd_detr", protocol: str = "release_round2s",
        output_root: Path | None = None,
    ) -> dict:
        output_root = output_root or self.root / "outputs" / name
        submission = output_root / "submission.jsonl"
        metric_output = output_root / "metrics.json"
        entry = {
            "name": name,
            "role": role,
            "backbone": backbone,
            "protocol": protocol,
            "checkpoint_roles": {"model": str(checkpoint)},
            "validation_metrics": str(metrics),
            "primary_metric_step": "evaluate",
            "expected_output_dir": str(output_root),
            "execution_steps": [{
                "id": "evaluate",
                "kind": "evaluate",
                "argv": [
                    sys.executable, "-m", "methods.qd_detr_gmr.evaluate",
                    "--checkpoint", str(checkpoint),
                    "--eval_annotation", str(self.test_annotation),
                    "--submission_path", str(submission),
                    "--metrics_path", str(metric_output),
                    "--no-diagnostic_decoders",
                ],
                "annotation_flag": "--eval_annotation",
                "checkpoint_bindings": {"--checkpoint": "model"},
                "input_bindings": {},
                "output_bindings": {
                    "--submission_path": {"output": "submission"},
                    "--metrics_path": {"output": "metrics"},
                },
                "outputs": {
                    "submission": str(submission),
                    "metrics": str(metric_output),
                },
                "depends_on": [],
                "environment": {"CUDA_VISIBLE_DEVICES": "0"},
            }],
        }
        if role == "candidate":
            entry.update({
                "reference_entry": "anchor",
                "group_diagnostics": str(self.diagnostics),
            })
        return entry

    def base_spec(self) -> dict:
        return {
            "schema_version": 2,
            "protocol_id": "unit-matrix-v1",
            "max_executions": 1,
            "selection_split": "validation",
            "working_directory": str(self.root),
            "ledger_dir": str(self.root / "ledgers"),
            "validation_annotations": str(self.validation),
            "test_annotations": str(self.test_annotation),
            "entries": [
                self.qd_entry(
                    "anchor", "anchor", self.anchor_checkpoint, self.anchor_metrics
                ),
                self.qd_entry(
                    "candidate", "candidate", self.candidate_checkpoint,
                    self.candidate_metrics,
                ),
            ],
        }

    def freeze(self, spec: dict | None = None) -> dict:
        write_json(self.spec_path, spec or self.base_spec())
        return build_manifest(
            self.spec_path, self.test_annotation, [str(self.source)]
        )

    def freeze_to_disk(self, spec: dict | None = None) -> dict:
        manifest = self.freeze(spec)
        write_manifest_exclusive(manifest, self.manifest_path)
        return manifest

    @staticmethod
    def successful_qd_runner(calls: list[list[str]]):
        def run(argv, **kwargs):
            calls.append(list(argv))
            for flag in ("--submission_path", "--metrics_path"):
                path = Path(argv[argv.index(flag) + 1])
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}\n", encoding="utf-8")
            return subprocess.CompletedProcess(argv, 0, b"ok", b"")
        return run

    def test_freezes_dag_bindings_hashes_reference_and_gates(self) -> None:
        manifest = self.freeze()
        verify_manifest_seal(manifest)
        self.assertEqual(manifest["schema_version"], 2)
        self.assertEqual(manifest["max_executions"], 1)
        candidate = manifest["entries"][1]
        self.assertEqual(candidate["reference_entry"], "anchor")
        self.assertTrue(all(candidate["validation_gates"].values()))
        self.assertEqual(
            candidate["execution_steps"][0]["checkpoint_bindings"],
            {"--checkpoint": "model"},
        )
        self.assertEqual(len(candidate["checkpoint_roles"]["model"]["sha256"]), 64)

    def test_rejects_missing_or_nonfinite_required_metric(self) -> None:
        for value in (
            {"brief": {"mAP": 9.0, "G-mIoU@3": 31.0}},
            {"brief": {"mAP": float("inf"), "G-mIoU@3": 31.0, "mR+@5": 1.0}},
        ):
            with self.subTest(value=value):
                write_json(self.candidate_metrics, value)
                with self.assertRaisesRegex(ValueError, "required metric|finite"):
                    self.freeze()

    def test_rejects_candidate_reference_with_different_backbone_or_protocol(self) -> None:
        for field, value in (("backbone", "cg_detr"), ("protocol", "clean_mask")):
            with self.subTest(field=field):
                spec = self.base_spec()
                spec["entries"][1][field] = value
                with self.assertRaisesRegex(ValueError, f"reference {field} differs"):
                    self.freeze(spec)

    def test_rejects_candidate_reference_that_is_not_an_anchor(self) -> None:
        spec = self.base_spec()
        spec["entries"][0]["role"] = "diagnostic"
        with self.assertRaisesRegex(ValueError, "must name an anchor"):
            self.freeze(spec)

    def test_rejects_checkpoint_path_not_bound_to_its_role(self) -> None:
        spec = self.base_spec()
        step = spec["entries"][1]["execution_steps"][0]
        index = step["argv"].index("--checkpoint") + 1
        step["argv"][index] = str(self.anchor_checkpoint)
        with self.assertRaisesRegex(ValueError, "wrong path"):
            self.freeze(spec)

    def test_rejects_annotation_as_unbound_or_wrong_flag(self) -> None:
        spec = self.base_spec()
        step = spec["entries"][1]["execution_steps"][0]
        step["annotation_flag"] = "--checkpoint"
        with self.assertRaisesRegex(ValueError, "annotation_flag"):
            self.freeze(spec)

        spec = self.base_spec()
        step = spec["entries"][1]["execution_steps"][0]
        step["argv"][step["argv"].index("--eval_annotation") + 1] = str(self.validation)
        step["argv"].append(str(self.test_annotation))
        with self.assertRaisesRegex(ValueError, "not bound"):
            self.freeze(spec)

    def test_rejects_unknown_dependency_cycle_and_unrelated_upstream_output(self) -> None:
        spec = self.base_spec()
        spec["entries"][1]["execution_steps"][0]["depends_on"] = ["missing"]
        with self.assertRaisesRegex(ValueError, "unknown dependency"):
            self.freeze(spec)

        moment = self.moment_entry()
        moment["execution_steps"][0]["depends_on"] = ["fuse"]
        spec = self.base_spec()
        spec["entries"] = [moment]
        with self.assertRaisesRegex(ValueError, "cycle"):
            self.freeze(spec)

        moment = self.moment_entry()
        fuse = next(step for step in moment["execution_steps"] if step["id"] == "fuse")
        fuse["depends_on"] = ["localization"]
        spec["entries"] = [moment]
        with self.assertRaisesRegex(ValueError, "must come from a dependency"):
            self.freeze(spec)

    def test_rejects_overlapping_or_nonpristine_output_roots(self) -> None:
        spec = self.base_spec()
        anchor_root = Path(spec["entries"][0]["expected_output_dir"])
        nested = anchor_root / "candidate"
        candidate = self.qd_entry(
            "candidate", "candidate", self.candidate_checkpoint,
            self.candidate_metrics, output_root=nested,
        )
        spec["entries"][1] = candidate
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.freeze(spec)

        spec = self.base_spec()
        root = Path(spec["entries"][0]["expected_output_dir"])
        (root / "empty-child").mkdir(parents=True)
        with self.assertRaisesRegex(RuntimeError, "not pristine"):
            self.freeze(spec)

    def test_qd_diagnostic_outputs_are_bound_as_deterministic_derivatives(self) -> None:
        spec = self.base_spec()
        step = spec["entries"][1]["execution_steps"][0]
        step["argv"].remove("--no-diagnostic_decoders")
        step["argv"].append("--diagnostic_decoders")
        submission = Path(step["outputs"]["submission"])
        metrics = Path(step["outputs"]["metrics"])
        step["outputs"].update({
            "threshold_submission": str(
                submission.with_name(f"{submission.stem}.threshold{submission.suffix}")
            ),
            "adaptive_submission": str(
                submission.with_name(f"{submission.stem}.adaptive{submission.suffix}")
            ),
            "threshold_metrics": str(
                metrics.with_name(f"{metrics.stem}.threshold{metrics.suffix}")
            ),
            "adaptive_metrics": str(
                metrics.with_name(f"{metrics.stem}.adaptive{metrics.suffix}")
            ),
        })
        step["output_bindings"]["--submission_path"]["derived_outputs"] = [
            "threshold_submission", "adaptive_submission",
        ]
        step["output_bindings"]["--metrics_path"]["derived_outputs"] = [
            "threshold_metrics", "adaptive_metrics",
        ]
        manifest = self.freeze(spec)
        frozen = manifest["entries"][1]["execution_steps"][0]
        self.assertEqual(len(frozen["outputs"]), 6)

        spec["entries"][1]["execution_steps"][0]["outputs"][
            "adaptive_metrics"
        ] = str(metrics.parent / "arbitrary.json")
        with self.assertRaisesRegex(ValueError, "naming contract"):
            self.freeze(spec)

        spec = self.base_spec()
        spec["entries"][1]["execution_steps"][0]["argv"].remove(
            "--no-diagnostic_decoders"
        )
        with self.assertRaisesRegex(ValueError, "explicitly select"):
            self.freeze(spec)

    def test_manifest_is_write_once_and_sealed(self) -> None:
        manifest = self.freeze()
        write_manifest_exclusive(manifest, self.manifest_path)
        with self.assertRaises(FileExistsError):
            write_manifest_exclusive(manifest, self.manifest_path)

        value = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        value["entries"][0]["execution_steps"][0]["argv"].append("--tampered")
        self.manifest_path.write_text(json.dumps(value), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "seal"):
            run_manifest(self.manifest_path, subprocess_runner=lambda *args, **kwargs: None)
        self.assertFalse((self.root / "ledgers" / "unit-matrix-v1.ledger.jsonl").exists())

    def test_runner_executes_exact_steps_once_and_writes_valid_ledger(self) -> None:
        manifest = self.freeze_to_disk()
        calls: list[list[str]] = []
        result = run_manifest(
            self.manifest_path,
            subprocess_runner=self.successful_qd_runner(calls),
        )
        self.assertEqual(result["status"], "completed")
        self.assertEqual(len(calls), 2)
        events = read_and_verify_ledger(manifest["ledger_path"])
        self.assertEqual(events[0]["event"], "matrix_started")
        self.assertEqual(events[-1]["event"], "matrix_completed")
        with self.assertRaises(AlreadyExecutedError):
            run_manifest(
                self.manifest_path,
                subprocess_runner=self.successful_qd_runner(calls),
            )
        self.assertEqual(len(calls), 2)

    def test_ledger_hash_chain_detects_rewrite(self) -> None:
        manifest = self.freeze_to_disk()
        calls: list[list[str]] = []
        run_manifest(
            self.manifest_path,
            subprocess_runner=self.successful_qd_runner(calls),
        )
        ledger_path = Path(manifest["ledger_path"])
        lines = ledger_path.read_text(encoding="utf-8").splitlines()
        event = json.loads(lines[1])
        event["event"] = "rewritten"
        lines[1] = json.dumps(event)
        ledger_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "event hash"):
            read_and_verify_ledger(ledger_path)

    def test_preclaim_hash_change_stops_before_any_execution(self) -> None:
        manifest = self.freeze_to_disk()
        original = self.candidate_checkpoint.read_bytes()
        self.candidate_checkpoint.write_bytes(b"X" * len(original))
        calls: list[list[str]] = []
        with self.assertRaisesRegex(RuntimeError, "artifact changed"):
            run_manifest(
                self.manifest_path,
                subprocess_runner=self.successful_qd_runner(calls),
            )
        self.assertEqual(calls, [])
        self.assertFalse(Path(manifest["ledger_path"]).exists())

    def test_source_inventory_addition_stops_before_execution(self) -> None:
        source_dir = self.root / "source_tree"
        source_dir.mkdir()
        (source_dir / "model.py").write_text("VALUE = 1\n", encoding="utf-8")
        write_json(self.spec_path, self.base_spec())
        manifest = build_manifest(
            self.spec_path, self.test_annotation, [str(source_dir)]
        )
        write_manifest_exclusive(manifest, self.manifest_path)
        (source_dir / "unregistered.py").write_text("VALUE = 2\n", encoding="utf-8")
        calls: list[list[str]] = []
        with self.assertRaisesRegex(RuntimeError, "source inventory changed"):
            run_manifest(
                self.manifest_path,
                subprocess_runner=self.successful_qd_runner(calls),
            )
        self.assertEqual(calls, [])
        self.assertFalse(Path(manifest["ledger_path"]).exists())

    def test_rehashes_static_inputs_before_every_step(self) -> None:
        manifest = self.freeze_to_disk()
        calls = []

        def mutate_after_first(argv, **kwargs):
            calls.append(list(argv))
            for flag in ("--submission_path", "--metrics_path"):
                path = Path(argv[argv.index(flag) + 1])
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}\n", encoding="utf-8")
            if len(calls) == 1:
                original = self.candidate_checkpoint.read_bytes()
                self.candidate_checkpoint.write_bytes(b"Z" * len(original))
            return subprocess.CompletedProcess(argv, 0, b"", b"")

        with self.assertRaisesRegex(RuntimeError, "artifact changed"):
            run_manifest(self.manifest_path, subprocess_runner=mutate_after_first)
        self.assertEqual(len(calls), 1)
        self.assertTrue(Path(manifest["ledger_path"]).exists())

    def test_command_failure_consumes_one_shot_claim(self) -> None:
        manifest = self.freeze_to_disk()
        calls = []

        def fail(argv, **kwargs):
            calls.append(list(argv))
            return subprocess.CompletedProcess(argv, 3, b"", b"failure")

        with self.assertRaises(RegisteredCommandError):
            run_manifest(self.manifest_path, subprocess_runner=fail)
        self.assertEqual(len(calls), 1)
        events = read_and_verify_ledger(manifest["ledger_path"])
        self.assertEqual(events[-1]["event"], "matrix_failed")
        with self.assertRaises(AlreadyExecutedError):
            run_manifest(self.manifest_path, subprocess_runner=fail)
        self.assertEqual(len(calls), 1)

    def test_missing_or_extra_output_fails_and_consumes_claim(self) -> None:
        for mode in ("missing", "extra"):
            with self.subTest(mode=mode):
                # Each subtest needs its own isolated one-shot protocol.
                protocol = f"unit-{mode}"
                spec = self.base_spec()
                spec["protocol_id"] = protocol
                for entry in spec["entries"]:
                    new_root = self.root / "outputs" / protocol / entry["name"]
                    entry["expected_output_dir"] = str(new_root)
                    step = entry["execution_steps"][0]
                    for output_name, old_path in list(step["outputs"].items()):
                        new_path = new_root / Path(old_path).name
                        step["outputs"][output_name] = str(new_path)
                    step["argv"][step["argv"].index("--submission_path") + 1] = step["outputs"]["submission"]
                    step["argv"][step["argv"].index("--metrics_path") + 1] = step["outputs"]["metrics"]
                manifest_path = self.root / f"{protocol}.manifest.json"
                manifest = self.freeze(spec)
                write_manifest_exclusive(manifest, manifest_path)

                def bad(argv, **kwargs):
                    submission = Path(argv[argv.index("--submission_path") + 1])
                    metrics = Path(argv[argv.index("--metrics_path") + 1])
                    submission.parent.mkdir(parents=True, exist_ok=True)
                    submission.write_text("{}\n", encoding="utf-8")
                    if mode == "extra":
                        metrics.write_text("{}\n", encoding="utf-8")
                        (submission.parent / "undeclared.txt").write_text("x", encoding="utf-8")
                    return subprocess.CompletedProcess(argv, 0, b"", b"")

                with self.assertRaises((FileNotFoundError, RuntimeError)):
                    run_manifest(manifest_path, subprocess_runner=bad)
                self.assertTrue(Path(manifest["ledger_path"]).exists())

    def moment_entry(self) -> dict:
        localization_checkpoint = self.root / "moment.localization.ckpt"
        decision_checkpoint = self.root / "moment.decision.ckpt"
        localization_checkpoint.write_bytes(b"localization")
        decision_checkpoint.write_bytes(b"decision")
        output_root = self.root / "outputs" / "moment"
        localization_dir = output_root / "localization"
        decision_dir = output_root / "decision"
        localization_submission = localization_dir / "moment_detr_gmr_test_submission.jsonl"
        decision_submission = decision_dir / "moment_detr_gmr_test_submission.jsonl"
        fused_submission = output_root / "fused" / "submission.jsonl"
        fused_manifest = output_root / "fused" / "fusion.json"
        fused_metrics = output_root / "fused" / "metrics.json"
        return {
            "name": "moment",
            "role": "anchor",
            "backbone": "moment_detr",
            "protocol": "release_round2s",
            "checkpoint_roles": {
                "localization": str(localization_checkpoint),
                "decision": str(decision_checkpoint),
            },
            "validation_metrics": str(self.anchor_metrics),
            "primary_metric_step": "fuse",
            "expected_output_dir": str(output_root),
            "execution_steps": [
                {
                    "id": "localization",
                    "kind": "predict",
                    "argv": [
                        sys.executable, "-m", "training.moment_detr_gmr.evaluate",
                        "--model_path", str(localization_checkpoint),
                        "--split", "test",
                        "--eval_path", str(self.test_annotation),
                        "--results_dir", str(localization_dir),
                    ],
                    "annotation_flag": "--eval_path",
                    "checkpoint_bindings": {"--model_path": "localization"},
                    "input_bindings": {},
                    "output_bindings": {
                        "--results_dir": {"directory": str(localization_dir)}
                    },
                    "outputs": {"submission": str(localization_submission)},
                    "depends_on": [],
                },
                {
                    "id": "decision",
                    "kind": "predict",
                    "argv": [
                        sys.executable, "-m", "training.moment_detr_gmr.evaluate",
                        "--model_path", str(decision_checkpoint),
                        "--split", "test",
                        "--eval_path", str(self.test_annotation),
                        "--results_dir", str(decision_dir),
                    ],
                    "annotation_flag": "--eval_path",
                    "checkpoint_bindings": {"--model_path": "decision"},
                    "input_bindings": {},
                    "output_bindings": {
                        "--results_dir": {"directory": str(decision_dir)}
                    },
                    "outputs": {"submission": str(decision_submission)},
                    "depends_on": [],
                },
                {
                    "id": "fuse",
                    "kind": "compose_evaluate",
                    "argv": [
                        sys.executable, "-m", "scripts.fuse_gmr_heads",
                        "--localization", str(localization_submission),
                        "--decision", str(decision_submission),
                        "--output", str(fused_submission),
                        "--manifest", str(fused_manifest),
                        "--ground-truth", str(self.test_annotation),
                        "--metrics-output", str(fused_metrics),
                    ],
                    "annotation_flag": "--ground-truth",
                    "checkpoint_bindings": {},
                    "input_bindings": {
                        "--localization": {"step": "localization", "output": "submission"},
                        "--decision": {"step": "decision", "output": "submission"},
                    },
                    "output_bindings": {
                        "--output": {"output": "submission"},
                        "--manifest": {"output": "fusion_manifest"},
                        "--metrics-output": {"output": "metrics"},
                    },
                    "outputs": {
                        "submission": str(fused_submission),
                        "fusion_manifest": str(fused_manifest),
                        "metrics": str(fused_metrics),
                    },
                    "depends_on": ["localization", "decision"],
                },
            ],
        }

    def test_moment_two_checkpoint_three_step_dag_and_role_binding(self) -> None:
        spec = self.base_spec()
        spec["entries"] = [self.moment_entry()]
        manifest = self.freeze(spec)
        entry = manifest["entries"][0]
        self.assertEqual(
            [step["id"] for step in entry["execution_steps"]],
            ["localization", "decision", "fuse"],
        )
        self.assertEqual(entry["primary_metric_step"], "fuse")
        self.assertFalse(entry["execution_steps"][0]["produces_primary_metrics"])
        self.assertTrue(entry["execution_steps"][2]["produces_primary_metrics"])

        spec = self.base_spec()
        moment = self.moment_entry()
        fuse = next(step for step in moment["execution_steps"] if step["id"] == "fuse")
        localization_index = fuse["argv"].index("--localization") + 1
        decision_index = fuse["argv"].index("--decision") + 1
        fuse["argv"][localization_index], fuse["argv"][decision_index] = (
            fuse["argv"][decision_index], fuse["argv"][localization_index]
        )
        spec["entries"] = [moment]
        with self.assertRaisesRegex(ValueError, "wrong upstream path"):
            self.freeze(spec)

        spec = self.base_spec()
        moment = self.moment_entry()
        localization = next(
            step for step in moment["execution_steps"] if step["id"] == "localization"
        )
        decision = next(
            step for step in moment["execution_steps"] if step["id"] == "decision"
        )
        localization["checkpoint_bindings"]["--model_path"] = "decision"
        decision["checkpoint_bindings"]["--model_path"] = "localization"
        localization["argv"][localization["argv"].index("--model_path") + 1] = (
            moment["checkpoint_roles"]["decision"]
        )
        decision["argv"][decision["argv"].index("--model_path") + 1] = (
            moment["checkpoint_roles"]["localization"]
        )
        spec["entries"] = [moment]
        with self.assertRaisesRegex(ValueError, "checkpoint role 'localization'"):
            self.freeze(spec)

    def test_moment_upstream_output_is_rehashed_before_fusion(self) -> None:
        spec = self.base_spec()
        spec["protocol_id"] = "unit-moment-rehash"
        spec["entries"] = [self.moment_entry()]
        manifest = self.freeze(spec)
        write_manifest_exclusive(manifest, self.manifest_path)
        calls = []
        localization_output = Path(
            manifest["entries"][0]["execution_steps"][0]["outputs"]["submission"]
        )

        def mutate_upstream(argv, **kwargs):
            calls.append(list(argv))
            results_dir = Path(argv[argv.index("--results_dir") + 1])
            output = results_dir / "moment_detr_gmr_test_submission.jsonl"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text("{}\n", encoding="utf-8")
            if len(calls) == 2:
                localization_output.write_text('{"tampered":true}\n', encoding="utf-8")
            return subprocess.CompletedProcess(argv, 0, b"", b"")

        with self.assertRaisesRegex(RuntimeError, "artifact changed"):
            run_manifest(self.manifest_path, subprocess_runner=mutate_upstream)
        self.assertEqual(len(calls), 2)
        self.assertTrue(Path(manifest["ledger_path"]).exists())


if __name__ == "__main__":
    unittest.main()
