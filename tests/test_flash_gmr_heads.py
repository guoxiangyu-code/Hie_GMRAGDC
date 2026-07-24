from __future__ import annotations

import unittest

import torch

from models.flash_vtg_gmr.gmr_heads import (
    FlashCandidateQualityHead,
    FlashIndependentZeroVerifier,
    flash_candidate_quality_loss,
    flash_independent_zero_loss,
    quality_calibrated_scores,
)


class FlashGMRHeadsTest(unittest.TestCase):
    def test_quality_head_and_non_null_quality_loss(self):
        torch.manual_seed(11)
        head = FlashCandidateQualityHead(hidden_dim=8, dropout=0.0)
        features = [torch.randn(2, 3, 8), torch.randn(2, 2, 8)]
        logits = head(features)
        self.assertEqual(tuple(logits.shape), (2, 5))
        # The second row is null and must contribute no quality supervision.
        spans = torch.tensor([
            [[0.10, 0.30], [0.20, 0.50], [0.55, 0.75], [0.00, 0.10], [0.80, 1.00]],
            [[0.10, 0.30], [0.20, 0.50], [0.55, 0.75], [0.00, 0.10], [0.80, 1.00]],
        ])
        targets = [
            {"spans": torch.tensor([[0.25, 0.30]])},  # cxw -> [0.10,0.40]
            {"spans": torch.zeros(0, 2)},
        ]
        loss = flash_candidate_quality_loss(logits, spans, targets)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(head.net[-1].weight.grad.norm()), 0.0)

    def test_independent_zero_verifier_and_loss(self):
        torch.manual_seed(13)
        verifier = FlashIndependentZeroVerifier(hidden_dim=8, dropout=0.0)
        zero_logits = verifier(
            query=torch.randn(3, 1, 8),
            video=torch.randn(3, 6, 8),
            video_mask=torch.tensor([
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 0, 0],
                [1, 1, 1, 0, 0, 0],
            ]),
            candidate_logits=torch.randn(3, 4, 1),
            candidate_spans_xx=torch.rand(3, 4, 2).sort(dim=-1).values,
            quality_logits=torch.randn(3, 4),
        )
        self.assertEqual(tuple(zero_logits.shape), (3,))
        loss = flash_independent_zero_loss(
            zero_logits,
            torch.tensor([1.0, 0.0, 1.0]),
            positive_query_weight=2.0,
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(verifier.classifier[-1].weight.grad.norm()), 0.0)

    def test_quality_score_calibration_preserves_shapes_and_ranking_signal(self):
        foreground = torch.tensor([[0.8, 0.7]])
        quality_logits = torch.tensor([[-3.0, 3.0]])
        calibrated = quality_calibrated_scores(
            foreground, quality_logits, alpha=0.5
        )
        self.assertEqual(tuple(calibrated.shape), (1, 2))
        self.assertGreater(float(calibrated[0, 1]), float(calibrated[0, 0]))
        self.assertTrue(torch.equal(
            quality_calibrated_scores(foreground, None), foreground
        ))


if __name__ == "__main__":
    unittest.main()
