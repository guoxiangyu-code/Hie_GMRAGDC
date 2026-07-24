from __future__ import annotations

import unittest

from methods.cg_detr_gmr.train import STAGE_NEW_PREFIXES as CG_PREFIXES
from methods.eatr_gmr.train import STAGE_NEW_PREFIXES as EATR_PREFIXES
from methods.qd_detr_gmr.train import STAGE_NEW_PREFIXES as QD_PREFIXES


class StagedOptimizerGroupingTest(unittest.TestCase):
    def test_gmr_stage_trains_only_the_new_existence_head_at_full_lr(self) -> None:
        self.assertEqual(QD_PREFIXES["qd_detr_gmr"], ("exist_head.",))
        self.assertEqual(EATR_PREFIXES["eatr_gmr"], ("exist_head.",))
        self.assertEqual(CG_PREFIXES["cg_detr_gmr"], ("exist_head.",))

    def test_component_stages_treat_parent_existence_as_shared(self) -> None:
        expected = {
            "quality": ("quality_embed.",),
            "counter": ("hierarchical_counter.",),
        }
        for suffix, prefixes in expected.items():
            self.assertEqual(QD_PREFIXES[f"qd_{suffix}"], prefixes)
            self.assertEqual(EATR_PREFIXES[f"eatr_{suffix}"], prefixes)
            self.assertEqual(CG_PREFIXES[f"cg_{suffix}"], prefixes)
            self.assertNotIn("exist_head.", prefixes)

        self.assertEqual(QD_PREFIXES["qd_dual"], ("dual_grounding.",))
        self.assertEqual(EATR_PREFIXES["eatr_dual"], ("dual_grounding.",))
        self.assertEqual(CG_PREFIXES["cg_phrase"], ("phrase_grounding.",))
        self.assertEqual(
            EATR_PREFIXES["eatr_quality_dual"],
            ("quality_embed.", "dual_grounding."),
        )

    def test_quality_grounding_stages_exclude_counter(self) -> None:
        self.assertEqual(
            QD_PREFIXES["qd_quality_dual"],
            ("quality_embed.", "dual_grounding."),
        )
        self.assertEqual(
            CG_PREFIXES["cg_quality_phrase"],
            ("quality_embed.", "phrase_grounding."),
        )
        self.assertNotIn("hierarchical_counter.", QD_PREFIXES["qd_quality_dual"])
        self.assertNotIn("hierarchical_counter.", CG_PREFIXES["cg_quality_phrase"])

    def test_full_stage_contains_all_and_only_new_hiea2m_modules(self) -> None:
        self.assertEqual(
            QD_PREFIXES["qd_hiea2m"],
            ("quality_embed.", "dual_grounding.", "hierarchical_counter."),
        )
        self.assertEqual(
            EATR_PREFIXES["eatr_hiea2m"],
            ("quality_embed.", "dual_grounding.", "hierarchical_counter."),
        )
        self.assertEqual(
            CG_PREFIXES["cg_hiea2m"],
            ("quality_embed.", "phrase_grounding.", "hierarchical_counter."),
        )


if __name__ == "__main__":
    unittest.main()
