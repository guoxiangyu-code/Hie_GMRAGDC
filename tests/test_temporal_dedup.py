from __future__ import annotations

import unittest

import torch

from models.moment_detr_gmr.temporal_dedup import temporal_deduplicate


class TemporalDedupTest(unittest.TestCase):
    def test_none_filters_ranks_and_preserves_inputs(self):
        spans = torch.tensor([[2.0, 3.0], [0.0, 1.0], [4.0, 5.0]])
        scores = torch.tensor([0.4, 0.9, 0.2])
        original_spans = spans.clone()
        original_scores = scores.clone()

        selected_spans, selected_scores, indices = temporal_deduplicate(
            spans,
            scores,
            method="none",
            score_threshold=0.3,
        )

        self.assertEqual(indices.tolist(), [1, 0])
        self.assertTrue(torch.equal(selected_spans, spans[indices]))
        self.assertTrue(torch.equal(selected_scores, scores[indices]))
        self.assertTrue(torch.equal(spans, original_spans))
        self.assertTrue(torch.equal(scores, original_scores))

    def test_hard_nms_removes_duplicate(self):
        spans = torch.tensor([[0.0, 1.0], [0.02, 1.02], [2.0, 3.0]])
        scores = torch.tensor([0.9, 0.8, 0.7])
        selected_spans, selected_scores, indices = temporal_deduplicate(
            spans,
            scores,
            method="hard_nms",
            iou_threshold=0.8,
        )
        self.assertEqual(indices.tolist(), [0, 2])
        self.assertTrue(torch.equal(selected_spans, spans[indices]))
        self.assertTrue(torch.equal(selected_scores, scores[indices]))

    def test_fixed_k_direct_topk_is_fair_baseline_for_dedup_ranking(self):
        # Raw Top-K spends both slots on the same event.  Soft de-duplication
        # uses the identical K but promotes the independent third event.
        spans = torch.tensor([[0.0, 1.0], [0.01, 1.01], [2.0, 3.0]])
        scores = torch.tensor([0.9, 0.85, 0.8])
        _, _, direct_indices = temporal_deduplicate(
            spans,
            scores,
            method="direct_topk",
            max_output=2,
        )
        _, _, dedup_indices = temporal_deduplicate(
            spans,
            scores,
            method="gaussian_soft_nms",
            soft_sigma=0.5,
            max_output=2,
        )
        self.assertEqual(direct_indices.tolist(), [0, 1])
        self.assertEqual(dedup_indices.tolist(), [0, 2])
        self.assertEqual(direct_indices.numel(), dedup_indices.numel())

    def test_explicit_distinctness_protects_identical_events(self):
        spans = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
        scores = torch.tensor([0.9, 0.8])
        protected = torch.tensor([[False, True], [False, False]])

        for method in ("hard_nms", "diou_nms", "gaussian_soft_nms"):
            with self.subTest(method=method):
                _, selected_scores, indices = temporal_deduplicate(
                    spans,
                    scores,
                    method=method,
                    iou_threshold=0.5,
                    diou_threshold=0.5,
                    soft_sigma=0.1,
                    protected_pairs=protected,
                )
                self.assertEqual(indices.tolist(), [0, 1])
                self.assertTrue(torch.equal(selected_scores, scores))

    def test_gaussian_soft_nms_is_gentler_with_larger_sigma(self):
        spans = torch.tensor([[0.0, 1.0], [0.05, 1.05]])
        scores = torch.tensor([0.9, 0.8])
        _, sharp_scores, sharp_indices = temporal_deduplicate(
            spans,
            scores,
            method="gaussian_soft_nms",
            soft_sigma=0.2,
        )
        _, gentle_scores, gentle_indices = temporal_deduplicate(
            spans,
            scores,
            method="gaussian_soft_nms",
            soft_sigma=2.0,
        )
        self.assertEqual(sharp_indices.tolist(), [0, 1])
        self.assertEqual(gentle_indices.tolist(), [0, 1])
        self.assertGreater(float(gentle_scores[1]), float(sharp_scores[1]))
        self.assertLess(float(gentle_scores[1]), float(scores[1]))

    def test_diou_preserves_shifted_overlap_that_iou_nms_suppresses(self):
        spans = torch.tensor([[0.0, 1.0], [0.1, 1.1]])
        scores = torch.tensor([0.9, 0.8])
        _, _, hard_indices = temporal_deduplicate(
            spans,
            scores,
            method="hard_nms",
            iou_threshold=0.815,
        )
        _, _, diou_indices = temporal_deduplicate(
            spans,
            scores,
            method="diou_nms",
            diou_threshold=0.815,
        )
        self.assertEqual(hard_indices.tolist(), [0])
        self.assertEqual(diou_indices.tolist(), [0, 1])

    def test_geometry_protection_preserves_shifted_events(self):
        spans = torch.tensor([[0.0, 1.0], [0.1, 1.1]])
        scores = torch.tensor([0.9, 0.8])
        _, _, indices = temporal_deduplicate(
            spans,
            scores,
            method="hard_nms",
            iou_threshold=0.5,
            protect_center_distance=0.09,
        )
        self.assertEqual(indices.tolist(), [0, 1])

    def test_cluster_vote_fuses_duplicates_but_keeps_protected_event(self):
        spans = torch.tensor(
            [[0.0, 1.0], [0.02, 1.02], [0.08, 1.08]], dtype=torch.float64
        )
        scores = torch.tensor([0.9, 0.8, 0.7], dtype=torch.float64)
        protected = torch.zeros(3, 3, dtype=torch.bool)
        protected[0, 2] = True
        protected[1, 2] = True

        selected_spans, selected_scores, indices = temporal_deduplicate(
            spans,
            scores,
            method="cluster_vote_soft_nms",
            cluster_iou_threshold=0.9,
            cluster_center_distance_threshold=0.1,
            cluster_duration_ratio_threshold=0.9,
            soft_sigma=0.2,
            protected_pairs=protected,
        )

        self.assertEqual(indices.tolist(), [0, 2])
        expected_start = (0.9 * 1.0 * 0.0 + 0.8 * (0.98 / 1.02) * 0.02) / (
            0.9 * 1.0 + 0.8 * (0.98 / 1.02)
        )
        self.assertAlmostEqual(float(selected_spans[0, 0]), expected_start, places=7)
        self.assertAlmostEqual(float(selected_spans[0, 1]), 1.0 + expected_start, places=7)
        self.assertTrue(torch.equal(selected_scores, torch.tensor([0.9, 0.7], dtype=torch.float64)))

    def test_score_threshold_and_max_output_apply_after_soft_decay(self):
        spans = torch.tensor([[0.0, 1.0], [0.01, 1.01], [2.0, 3.0]])
        scores = torch.tensor([0.9, 0.8, 0.7])
        _, selected_scores, indices = temporal_deduplicate(
            spans,
            scores,
            method="gaussian_soft_nms",
            soft_sigma=0.1,
            score_threshold=0.2,
            max_output=2,
        )
        self.assertEqual(indices.tolist(), [0, 2])
        self.assertTrue(torch.allclose(selected_scores, torch.tensor([0.9, 0.7])))

    def test_empty_input_and_zero_max_output(self):
        spans = torch.empty(0, 2)
        scores = torch.empty(0)
        for method in (
            "none",
            "direct_topk",
            "hard_nms",
            "gaussian_soft_nms",
            "diou_nms",
            "cluster_vote_soft_nms",
        ):
            with self.subTest(method=method):
                selected_spans, selected_scores, indices = temporal_deduplicate(
                    spans, scores, method=method
                )
                self.assertEqual(tuple(selected_spans.shape), (0, 2))
                self.assertEqual(tuple(selected_scores.shape), (0,))
                self.assertEqual(indices.dtype, torch.long)

        selected_spans, selected_scores, indices = temporal_deduplicate(
            torch.tensor([[0.0, 1.0]]),
            torch.tensor([0.9]),
            max_output=0,
        )
        self.assertEqual(selected_spans.numel(), 0)
        self.assertEqual(selected_scores.numel(), 0)
        self.assertEqual(indices.numel(), 0)

    def test_invalid_inputs_and_parameters_are_rejected(self):
        spans = torch.tensor([[0.0, 1.0]])
        scores = torch.tensor([0.9])
        with self.assertRaisesRegex(ValueError, "unknown temporal"):
            temporal_deduplicate(spans, scores, method="mystery")
        with self.assertRaisesRegex(ValueError, "positive duration"):
            temporal_deduplicate(torch.tensor([[1.0, 1.0]]), scores)
        with self.assertRaisesRegex(ValueError, "shape"):
            temporal_deduplicate(spans, scores, protected_pairs=torch.zeros(2, 2))
        with self.assertRaisesRegex(ValueError, "soft_sigma"):
            temporal_deduplicate(spans, scores, soft_sigma=0.0)


if __name__ == "__main__":
    unittest.main()
