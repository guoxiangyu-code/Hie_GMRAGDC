from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.ablate_temporal_dedup import (
    assert_validation_input_path,
    build_parser,
    cluster_temporal_windows,
    complete_link_clusters,
    diou_temporal_nms,
    hard_temporal_nms,
    run_ablation,
    soft_temporal_nms,
    temporal_iou,
    temporal_diou,
)


class TemporalDedupAlgorithmsTest(unittest.TestCase):
    def test_temporal_iou_and_hard_nms(self):
        self.assertAlmostEqual(temporal_iou([0, 10], [2, 12]), 8 / 12)
        windows = [[0, 10, 0.9], [1, 9, 0.8], [20, 30, 0.7]]
        selected = hard_temporal_nms(windows, iou_threshold=0.5)
        self.assertEqual(selected, [[0, 10, 0.9], [20, 30, 0.7]])

    def test_linear_soft_nms_decays_instead_of_hard_deleting(self):
        windows = [[0, 10, 0.9], [1, 9, 0.8], [20, 30, 0.7]]
        selected = soft_temporal_nms(
            windows,
            mode="linear",
            iou_threshold=0.5,
            score_floor=0.0,
        )
        self.assertEqual(selected[0], [0, 10, 0.9])
        self.assertEqual(selected[1], [20, 30, 0.7])
        self.assertAlmostEqual(selected[2][2], 0.16)

    def test_diou_nms_preserves_shifted_overlap(self):
        windows = [[0.0, 1.0, 0.9], [0.1, 1.1, 0.8]]
        self.assertLess(temporal_diou(windows[0], windows[1]), temporal_iou(windows[0], windows[1]))
        self.assertEqual(len(hard_temporal_nms(windows, iou_threshold=0.815)), 1)
        self.assertEqual(len(diou_temporal_nms(windows, threshold=0.815)), 2)

    def test_complete_link_avoids_transitive_chain(self):
        # A overlaps B, B overlaps C, but A does not overlap C above 0.5.
        windows = [[0, 10, 0.9], [2, 12, 0.8], [4, 14, 0.7]]
        clusters = complete_link_clusters(windows, iou_threshold=0.5)
        self.assertEqual([len(cluster) for cluster in clusters], [2, 1])

    def test_cluster_representative_and_fusion(self):
        windows = [[0, 10, 0.9], [2, 12, 0.8], [20, 30, 0.7]]
        representative = cluster_temporal_windows(
            windows, iou_threshold=0.5, output="representative"
        )
        self.assertEqual(representative, [[0, 10, 0.9], [20, 30, 0.7]])

        fused = cluster_temporal_windows(windows, iou_threshold=0.5, output="fusion")
        self.assertAlmostEqual(fused[0][0], 1.6 / 1.7)
        self.assertAlmostEqual(fused[0][1], 18.6 / 1.7)
        self.assertEqual(fused[0][2], 0.9)


class TemporalDedupValidationGuardTest(unittest.TestCase):
    def test_rejects_test_before_file_access(self):
        with self.assertRaisesRegex(ValueError, "test-labelled"):
            assert_validation_input_path("does_not_exist_test.jsonl", role="ground truth")

    def test_requires_explicit_validation_token(self):
        with self.assertRaisesRegex(ValueError, "Validation-only guard"):
            assert_validation_input_path("ground_truth.jsonl", role="ground truth")
        self.assertEqual(
            assert_validation_input_path("pred_val.jsonl", role="prediction"),
            Path("pred_val.jsonl"),
        )


class TemporalDedupEndToEndTest(unittest.TestCase):
    @staticmethod
    def _write_jsonl(path: Path, rows):
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")

    def test_writes_predictions_metrics_and_summary(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            prediction_path = root / "tiny_val_predictions.jsonl"
            gt_path = root / "tiny_val_gt.jsonl"
            output_dir = root / "outputs"

            predictions = [
                {
                    "qid": 1,
                    "pred_relevant_windows": [
                        [0, 10, 0.9],
                        [1, 9, 0.8],
                        [20, 30, 0.7],
                    ],
                    "pred_exist_score": 0.9,
                    "pred_count": 2,
                    "pred_count_probs": [0.01, 0.09, 0.8, 0.05, 0.05],
                },
                {
                    "qid": 2,
                    "pred_relevant_windows": [[40, 50, 0.2]],
                    "pred_exist_score": 0.1,
                    "pred_count": 0,
                    "pred_count_probs": [0.9, 0.025, 0.025, 0.025, 0.025],
                },
            ]
            ground_truth = [
                {"qid": 1, "relevant_windows": [[0, 10], [20, 30]]},
                {"qid": 2, "relevant_windows": []},
            ]
            self._write_jsonl(prediction_path, predictions)
            self._write_jsonl(gt_path, ground_truth)

            args = build_parser().parse_args([
                "--prediction-path", str(prediction_path),
                "--gt-path", str(gt_path),
                "--output-dir", str(output_dir),
                "--methods", "none", "hard_nms",
                "--iou-thresholds", "0.5",
                "--map-num-workers", "1",
            ])
            summary = run_ablation(args)

            # Two ranking methods x mandatory Top-1/3/5/predicted-count budgets.
            self.assertEqual(len(summary["methods"]), 8)
            self.assertFalse(summary["protocol"]["test_labels_read"])
            self.assertTrue(summary["prediction_schema_audit"]["supports_geometry_score_dedup"])
            self.assertFalse(summary["prediction_schema_audit"]["supports_candidate_semantic_dedup"])
            for method in summary["methods"]:
                self.assertTrue(Path(method["prediction_output"]).is_file())
                self.assertTrue(Path(method["metrics_output"]).is_file())
                self.assertIn("mAP", method["brief"])
                self.assertIn("delta_vs_direct_topk", method)
                self.assertEqual(method["delta_vs_direct_topk"]["AUROC"], 0.0)
                with Path(method["prediction_output"]).open("r", encoding="utf-8") as handle:
                    output_rows = [json.loads(line) for line in handle]
                self.assertEqual(
                    [row["pred_exist_score"] for row in output_rows],
                    [row["pred_exist_score"] for row in predictions],
                )
                with Path(method["metrics_output"]).open("r", encoding="utf-8") as handle:
                    metric_payload = json.load(handle)
                self.assertFalse(
                    metric_payload["dedup_ablation"]["existence_score_or_gate_changed"]
                )
            self.assertTrue((output_dir / "dedup_ablation_summary.json").is_file())


if __name__ == "__main__":
    unittest.main()
