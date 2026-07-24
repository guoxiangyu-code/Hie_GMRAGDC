from __future__ import annotations

import unittest

import torch
from easydict import EasyDict

from models.moment_detr_gmr.moment_detr import build_model
from training.moment_detr_gmr.train import MOMENT_VARIANT_FLAGS


def tiny_options(variant: str) -> EasyDict:
    (
        use_exist_head,
        use_quality_head,
        use_dual_grounding,
        use_hierarchical_counter,
        use_independent_zero_head,
        use_pairwise_head,
    ) = MOMENT_VARIANT_FLAGS[variant]
    return EasyDict(
        device="cpu",
        hidden_dim=16,
        dropout=0.0,
        nheads=4,
        dim_feedforward=32,
        enc_layers=1,
        dec_layers=1,
        position_embedding="sine",
        max_q_l=6,
        max_v_l=8,
        input_dropout=0.0,
        t_feat_dim=8,
        v_feat_dim=12,
        a_feat_dim=0,
        aux_loss=False,
        num_queries=4,
        span_loss_type="l1",
        n_input_proj=1,
        set_cost_class=4,
        set_cost_span=10,
        set_cost_giou=1,
        span_loss_coef=10,
        giou_loss_coef=1,
        label_loss_coef=4,
        lw_saliency=0,
        eos_coef=0.1,
        saliency_margin=0.2,
        use_exist_head=use_exist_head,
        exist_pool="max",
        use_quality_head=use_quality_head,
        use_dual_grounding=use_dual_grounding,
        dual_num_phrases=3,
        dual_num_dummies=3,
        dual_slot_iterations=1,
        dual_gate_init=-4.0,
        dual_nheads=4,
        use_hierarchical_counter=use_hierarchical_counter,
        use_independent_zero_head=use_independent_zero_head,
        use_pairwise_head=use_pairwise_head,
        mask_null_vmr_loss=True,
    )


def model_inputs() -> dict[str, torch.Tensor]:
    torch.manual_seed(71)
    return {
        "src_vid": torch.randn(2, 8, 12),
        "src_vid_mask": torch.tensor([
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 0, 0],
        ], dtype=torch.float32),
        "src_txt": torch.randn(2, 6, 8),
        "src_txt_mask": torch.tensor([
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 0],
        ], dtype=torch.float32),
    }


class MomentQualityDualVariantTest(unittest.TestCase):
    def test_quality_dual_structure_and_losses_exclude_counter(self):
        flags = MOMENT_VARIANT_FLAGS["md_quality_dual"]
        self.assertEqual(flags, (True, True, True, False, False, False))

        model, criterion = build_model(tiny_options("md_quality_dual"))
        self.assertIsNotNone(model.exist_head)
        self.assertIsNotNone(model.quality_embed)
        self.assertIsNotNone(model.dual_grounding)
        self.assertIsNone(model.hierarchical_counter)
        self.assertIn("exist", criterion.losses)
        self.assertIn("quality", criterion.losses)
        self.assertIn("dual", criterion.losses)
        self.assertNotIn("counter", criterion.losses)
        self.assertFalse(any("count" in name for name in criterion.weight_dict))

        outputs = model(**model_inputs())
        self.assertIn("pred_exist_logits", outputs)
        self.assertIn("pred_quality_logits", outputs)
        self.assertIn("dual_phrase_attention", outputs)
        self.assertNotIn("pred_positive_count_logits", outputs)

    def test_gmr_parent_warm_start_is_exact_for_parent_outputs(self):
        torch.manual_seed(73)
        parent, _ = build_model(tiny_options("md_gmr"))
        child, _ = build_model(tiny_options("md_quality_dual"))
        incompatible = child.load_state_dict(parent.state_dict(), strict=False)
        self.assertFalse(incompatible.unexpected_keys)
        self.assertTrue(incompatible.missing_keys)
        self.assertTrue(all(
            name.startswith(("quality_embed.", "dual_grounding."))
            for name in incompatible.missing_keys
        ))

        parent.eval()
        child.eval()
        inputs = model_inputs()
        with torch.no_grad():
            parent_outputs = parent(**inputs)
            child_outputs = child(**inputs)
        for name in ("pred_logits", "pred_spans", "pred_exist_logits"):
            self.assertTrue(
                torch.equal(parent_outputs[name], child_outputs[name]),
                msg=f"warm-start changed {name}",
            )


if __name__ == "__main__":
    unittest.main()
