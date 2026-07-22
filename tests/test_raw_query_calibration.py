from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from methods.cg_detr_gmr import engine as cg_engine
from methods.cg_detr_gmr.evaluate import build_parser as build_cg_parser
from methods.eatr_gmr.runtime import predict_views
from methods.qd_detr_gmr import engine as qd_engine
from methods.qd_detr_gmr.evaluate import build_parser as build_qd_parser
from scripts import calibrate_hiea2m as calibration
from scripts.calibrate_hiea2m import (
    build_annotation_provenance,
    build_parser,
    build_producer_provenance,
    decode_submission,
    load_reference_metrics,
    sha256,
    validate_qid_coverage,
)


class _SyntheticCounterModel:
    def eval(self):
        return self

    def __call__(self, **_inputs):
        return {
            "pred_logits": torch.tensor([[
                [3.0, 0.0],
                [1.0, 0.0],
                [2.0, 0.0],
            ]]),
            "pred_spans": torch.tensor([[
                [0.20, 0.20],
                [0.60, 0.20],
                [0.85, 0.10],
            ]]),
            "pred_quality_logits": torch.tensor([[0.0, 2.0, -1.0]]),
            "pred_exist_logits": torch.tensor([2.0]),
            "pred_positive_count_logits": torch.tensor([[0.0, 5.0, 0.0, 0.0]]),
        }


def _metadata():
    return [{"qid": 7, "query": "synthetic query", "vid": "synthetic", "duration": 10.0}]


def _assert_standard_raw_row(test: unittest.TestCase, row: dict) -> None:
    test.assertEqual(row["pred_count"], 2)
    test.assertEqual(len(row["pred_count_probs"]), 5)
    test.assertAlmostEqual(sum(row["pred_count_probs"]), 1.0, places=5)
    windows = row["all_query_windows"]
    components = row["all_query_components"]
    test.assertEqual(len(windows), 3)
    test.assertEqual(len(components), 3)
    for window, component in zip(windows, components):
        test.assertEqual(len(window), 3)
        test.assertEqual(len(component), 2)
        expected_score = component[0] ** 0.5 * component[1] ** 0.5
        test.assertAlmostEqual(window[2], expected_score, places=5)


class RawQueryProducerTest(unittest.TestCase):
    def test_qd_and_cg_emit_the_same_standard_raw_contract(self):
        batch = (_metadata(), {}, {})
        with mock.patch.object(
            qd_engine, "prepare_batch", side_effect=lambda value, _device: value
        ):
            qd_row = qd_engine.predict_modes(
                _SyntheticCounterModel(),
                [batch],
                torch.device("cpu"),
                clip_length=2.0,
                round_to_clip=False,
                decode_modes=("full",),
                save_raw_queries=True,
            )["full"][0]
        with mock.patch.object(
            cg_engine, "prepare_batch", side_effect=lambda value, _device: value
        ):
            cg_row = cg_engine.predict(
                _SyntheticCounterModel(),
                [batch],
                torch.device("cpu"),
                clip_length=2.0,
                round_to_clip=False,
                save_raw_queries=True,
            )[0]

        _assert_standard_raw_row(self, qd_row)
        _assert_standard_raw_row(self, cg_row)
        self.assertEqual(qd_row["all_query_windows"], cg_row["all_query_windows"])
        self.assertEqual(qd_row["all_query_components"], cg_row["all_query_components"])
        self.assertNotIn("pred_count_probabilities", qd_row)
        self.assertNotIn("pred_count_probabilities", cg_row)

    def test_eatr_emits_aligned_windows_and_components(self):
        row = predict_views(
            _SyntheticCounterModel(),
            [(_metadata(), {}, {})],
            torch.device("cpu"),
            modes=("full",),
            round_to_clip=False,
            save_raw_queries=True,
        )["full"][0]
        _assert_standard_raw_row(self, row)

    def test_qd_and_cg_evaluate_parsers_expose_raw_query_flag(self):
        qd_args = build_qd_parser().parse_args([
            "--checkpoint", "synthetic.ckpt", "--save_raw_queries",
        ])
        cg_args = build_cg_parser().parse_args([
            "--checkpoint", "synthetic.ckpt", "--save_raw_queries",
        ])
        self.assertTrue(qd_args.save_raw_queries)
        self.assertTrue(cg_args.save_raw_queries)


class CalibrationSchemaTest(unittest.TestCase):
    config = {
        "decode_mode": "full",
        "existence_threshold": 0.4,
        "count_confidence_threshold": 0.55,
        "window_score_threshold": 0.1,
        "quality_alpha": 0.5,
        "diversity_lambda": 0.0,
    }
    count_probabilities = [0.1, 0.8, 0.05, 0.03, 0.02]

    def _decode(self, row):
        return decode_submission(
            [row],
            self.config,
            round_to_clip=False,
            clip_length=2.0,
            max_ts=10.0,
        )[0]

    def _eatr_row(self):
        return {
            "qid": 1,
            "pred_count_probs": self.count_probabilities,
            "all_query_windows": [[8.0, 10.0, 0.99], [2.0, 4.0, 0.01]],
            "all_query_components": [[0.81, 0.25], [0.36, 1.0]],
        }

    def test_moment_and_eatr_raw_layouts_decode_identically(self):
        moment_row = {
            "qid": 1,
            "pred_count_probs": self.count_probabilities,
            "all_query_components": [
                [8.0, 10.0, 0.81, 0.25],
                [2.0, 4.0, 0.36, 1.0],
            ],
            # Moment windows are independently ranking-ordered.  The 4-column
            # components above are authoritative and must not be zipped to them.
            "all_query_windows": [[2.0, 4.0, 0.01], [8.0, 10.0, 0.99]],
        }
        expected = [[2.0, 4.0, 0.6], [8.0, 10.0, 0.45]]
        moment = self._decode(moment_row)
        eatr = self._decode(self._eatr_row())
        self.assertEqual(moment["pred_relevant_windows"], expected)
        self.assertEqual(eatr["pred_relevant_windows"], expected)
        self.assertEqual(moment["pred_count"], 1)
        self.assertEqual(eatr["pred_count"], 1)

    def test_raw_shape_contract_is_strict(self):
        cases = []

        components_1d = self._eatr_row()
        components_1d["all_query_components"] = [0.81, 0.25]
        cases.append(("components-1d", components_1d, "all_query_components"))

        components_three_columns = self._eatr_row()
        components_three_columns["all_query_components"] = [[0.8, 0.2, 0.1]]
        cases.append(("components-Nx3", components_three_columns, "all_query_components"))

        missing_windows = self._eatr_row()
        del missing_windows["all_query_windows"]
        cases.append(("missing-windows", missing_windows, "all_query_windows"))

        windows_two_columns = self._eatr_row()
        windows_two_columns["all_query_windows"] = [[8.0, 10.0], [2.0, 4.0]]
        cases.append(("windows-Nx2", windows_two_columns, "all_query_windows"))

        row_mismatch = self._eatr_row()
        row_mismatch["all_query_windows"] = [[8.0, 10.0, 0.99]]
        cases.append(("row-mismatch", row_mismatch, "row mismatch"))

        non_finite = self._eatr_row()
        non_finite["all_query_components"][0][0] = float("nan")
        cases.append(("non-finite", non_finite, "non-finite"))

        wrong_count_shape = self._eatr_row()
        wrong_count_shape["pred_count_probs"] = [0.1, 0.2, 0.3, 0.4]
        cases.append(("count-shape", wrong_count_shape, "pred_count_probs"))

        for name, row, message in cases:
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, rf"qid=1.*{message}"):
                    self._decode(row)

    def test_qid_validation_rejects_missing_duplicates_and_mismatch(self):
        with self.assertRaisesRegex(ValueError, "submission row 0.*missing.*qid"):
            decode_submission(
                [{}], self.config, round_to_clip=False, clip_length=2.0, max_ts=10.0
            )

        duplicate_rows = [self._eatr_row(), copy.deepcopy(self._eatr_row())]
        with self.assertRaisesRegex(ValueError, "duplicate qid=1"):
            decode_submission(
                duplicate_rows,
                self.config,
                round_to_clip=False,
                clip_length=2.0,
                max_ts=10.0,
            )

        with self.assertRaisesRegex(ValueError, "ground truth.*duplicate qid=1"):
            validate_qid_coverage([{"qid": 1}], [{"qid": 1}, {"qid": 1}])

        with self.assertRaisesRegex(
            ValueError, r"Incomplete validation coverage:.*missing=\[2\].*extra=\[3\]"
        ):
            validate_qid_coverage(
                [{"qid": 1}, {"qid": 3}],
                [{"qid": 1}, {"qid": 2}],
            )


class CalibrationProvenanceTest(unittest.TestCase):
    def test_annotation_contract_uses_explicit_role_identity_and_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            # The basename is deliberately misleading: split authorization is
            # based on the explicit role and pinned bytes, never a substring.
            annotation = Path(directory) / "heldout_test_named_file.jsonl"
            annotation.write_text('{"qid": 1}\n', encoding="utf-8")
            digest = sha256(str(annotation))
            provenance = build_annotation_provenance(
                str(annotation),
                identity="soccer-gmr-standard-val-v1",
                role="validation",
                expected_sha256=digest,
            )
            self.assertEqual(provenance["role"], "validation")
            self.assertEqual(provenance["identity"], "soccer-gmr-standard-val-v1")
            self.assertEqual(provenance["sha256"], digest)

            with self.assertRaisesRegex(ValueError, "annotation_role.*validation"):
                build_annotation_provenance(
                    str(annotation),
                    identity="soccer-gmr-standard-test-v1",
                    role="test",
                    expected_sha256=digest,
                )
            with self.assertRaisesRegex(ValueError, "digest does not match"):
                build_annotation_provenance(
                    str(annotation),
                    identity="soccer-gmr-standard-val-v1",
                    role="validation",
                    expected_sha256="0" * 64,
                )

    def test_missing_or_malformed_producer_provenance_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "model.ckpt"
            checkpoint.write_bytes(b"checkpoint")
            argv = root / "argv.json"
            argv.write_text('{}\n', encoding="utf-8")
            source = root / "evaluate.py"
            source.write_text("# producer\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "non-empty JSON array"):
                build_producer_provenance(
                    str(checkpoint), str(argv), [str(source)]
                )
            argv.write_text('["python", "evaluate.py"]\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "producer_source_file"):
                build_producer_provenance(str(checkpoint), str(argv), [])
            with self.assertRaisesRegex(ValueError, "producer checkpoint"):
                build_producer_provenance(
                    str(root / "missing.ckpt"), str(argv), [str(source)]
                )

    def test_reference_metrics_must_be_a_finite_positive_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            metrics = Path(directory) / "reference.json"
            metrics.write_text(
                json.dumps({"brief": {"mAP": 8.0, "G-mIoU@3": 30.0}}),
                encoding="utf-8",
            )
            reference = load_reference_metrics(str(metrics))
            self.assertEqual(reference["values"], {"mAP": 8.0, "G-mIoU@3": 30.0})
            self.assertEqual(reference["artifact_sha256"], sha256(str(metrics)))

            metrics.write_text(
                json.dumps({"brief": {"mAP": 0.0, "G-mIoU@3": 30.0}}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "finite and positive"):
                load_reference_metrics(str(metrics))

    def test_manifest_binds_inputs_producer_full_grid_and_selected_submission(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_row = {
                "qid": 1,
                "pred_count_probs": [0.1, 0.8, 0.05, 0.03, 0.02],
                "all_query_windows": [[2.0, 4.0, 0.9]],
                "all_query_components": [[0.81, 1.0]],
            }
            submission = root / "raw.jsonl"
            submission.write_text(json.dumps(raw_row) + "\n", encoding="utf-8")
            ground_truth = root / "annotation.jsonl"
            ground_truth.write_text('{"qid": 1}\n', encoding="utf-8")
            checkpoint = root / "model.ckpt"
            checkpoint.write_bytes(b"checkpoint")
            producer_argv = root / "argv.json"
            producer_argv.write_text(
                json.dumps(["python", "evaluate.py", "--save_raw_queries"]),
                encoding="utf-8",
            )
            producer_source = root / "evaluate.py"
            producer_source.write_text("# exact producer source\n", encoding="utf-8")
            reference_metrics = root / "reference.json"
            reference_metrics.write_text(
                json.dumps({"brief": {"mAP": 8.0, "G-mIoU@3": 30.0}}),
                encoding="utf-8",
            )
            output = root / "calibration.json"

            args = build_parser().parse_args([
                "--submission", str(submission),
                "--ground_truth", str(ground_truth),
                "--annotation_identity", "soccer-gmr-standard-val-v1",
                "--annotation_role", "validation",
                "--annotation_sha256", sha256(str(ground_truth)),
                "--producer_checkpoint", str(checkpoint),
                "--producer_argv_json", str(producer_argv),
                "--producer_source_files", str(producer_source),
                "--reference_metrics", str(reference_metrics),
                "--output", str(output),
                "--modes", "full",
                "--existence_thresholds", "0.4",
                "--quality_alphas", "0.5",
                "--diversity_lambdas", "0.0",
            ])
            metrics = {"brief": {"mAP": 9.0, "G-mIoU@3": 31.0}}
            with mock.patch.object(
                calibration,
                "normalize_ground_truth",
                return_value=([{"qid": 1}], None),
            ), mock.patch.object(calibration, "evaluate_gmr", return_value=metrics):
                manifest = calibration.calibrate(args)

            self.assertEqual(manifest["contract_version"], 2)
            self.assertEqual(manifest["calibration_split"]["role"], "validation")
            self.assertEqual(manifest["ground_truth"]["sha256"], sha256(str(ground_truth)))
            self.assertEqual(manifest["input_submission"]["sha256"], sha256(str(submission)))
            self.assertEqual(
                manifest["producer"]["checkpoint"]["sha256"], sha256(str(checkpoint))
            )
            self.assertEqual(
                manifest["producer"]["argv"]["artifact_sha256"],
                sha256(str(producer_argv)),
            )
            self.assertEqual(
                manifest["producer"]["source_files"][0]["sha256"],
                sha256(str(producer_source)),
            )
            self.assertEqual(
                manifest["reference_metrics"]["artifact_sha256"],
                sha256(str(reference_metrics)),
            )
            self.assertEqual(manifest["grid_size"], 1)
            self.assertEqual(len(manifest["grid_records"]), 1)
            selected_path = manifest["selected_submission"]["path"]
            self.assertEqual(
                manifest["selected_submission"]["sha256"], sha256(selected_path)
            )
            saved = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(saved, manifest)


if __name__ == "__main__":
    unittest.main()
