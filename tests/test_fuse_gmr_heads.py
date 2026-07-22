from __future__ import annotations

import unittest

from scripts.fuse_gmr_heads import fuse_submissions


class FuseGMRHeadsTest(unittest.TestCase):
    def test_field_ownership_and_localization_order(self):
        localization = [
            {
                "qid": 2,
                "pred_relevant_windows": [[1.0, 2.0, 0.7]],
                "pred_exist_score": 0.1,
                "pred_count": 4,
            },
            {
                "qid": 1,
                "pred_relevant_windows": [[3.0, 4.0, 0.6]],
                "pred_exist_score": 0.2,
            },
        ]
        decision = [
            {"qid": 1, "pred_exist_score": 0.8, "pred_count": 1},
            {
                "qid": 2,
                "pred_exist_score": 0.9,
                "pred_count": 2,
                "pred_count_probs": [0.1, 0.2, 0.6, 0.05, 0.05],
            },
        ]
        fused = fuse_submissions(localization, decision)
        self.assertEqual([row["qid"] for row in fused], [2, 1])
        self.assertEqual(fused[0]["pred_relevant_windows"], [[1.0, 2.0, 0.7]])
        self.assertEqual(fused[0]["pred_exist_score"], 0.9)
        self.assertEqual(fused[0]["pred_count"], 2)
        self.assertNotIn("pred_count_probs", fused[1])

    def test_coverage_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            fuse_submissions(
                [{"qid": 1, "pred_relevant_windows": []}],
                [{"qid": 2, "pred_exist_score": 0.5}],
            )


if __name__ == "__main__":
    unittest.main()
