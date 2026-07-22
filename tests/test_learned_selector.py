from __future__ import annotations

import unittest

from easydict import EasyDict

import torch

from models.moment_detr_gmr.learned_selector import (
    IndependentZeroVerifier,
    PairwiseSameEventHead,
    cautious_complete_link_fusion,
    combine_two_stage_existence,
    independent_zero_loss,
    learned_mmr_select,
    pairwise_same_event_loss,
    two_stage_accept,
)
from models.moment_detr_gmr.moment_detr import build_model


class LearnedSelectorTest(unittest.TestCase):
    @staticmethod
    def _model_args():
        return EasyDict(
            device="cpu", hidden_dim=16, dropout=0.0, nheads=4,
            dim_feedforward=32, enc_layers=1, dec_layers=1,
            pre_norm=False, position_embedding="sine", max_q_l=8,
            max_v_l=6, span_loss_type="l1", t_feat_dim=8, v_feat_dim=8,
            a_feat_dim=0, aux_loss=False, num_queries=4, input_dropout=0.0,
            n_input_proj=1, use_txt_pos=False, use_exist_head=True,
            exist_pool="max", use_dual_grounding=False,
            use_hierarchical_counter=True, counter_dropout=0.0,
            counter_detach_scores=True, use_quality_head=True,
            use_independent_zero_head=True, use_pairwise_head=True,
            selector_dropout=0.0, pairwise_detach_inputs=True,
            set_cost_span=1.0, set_cost_giou=1.0, set_cost_class=1.0,
            span_loss_coef=1.0, giou_loss_coef=1.0, label_loss_coef=1.0,
            lw_saliency=0.0, eos_coef=0.1, saliency_margin=0.2,
            exist_loss_coef=1.0, quality_loss_coef=1.0,
            count_loss_coef=1.0, count_ordinal_loss_coef=0.25,
            count_contrastive_loss_coef=0.05, count_consistency_loss_coef=0.05,
            zero_loss_coef=1.0, pairwise_loss_coef=1.0,
            mask_null_vmr_loss=True,
        )

    def test_full_model_emits_independent_and_pairwise_outputs(self):
        model, _ = build_model(self._model_args())
        output = model(
            src_txt=torch.randn(2, 5, 8),
            src_txt_mask=torch.ones(2, 5),
            src_vid=torch.randn(2, 6, 8),
            src_vid_mask=torch.ones(2, 6),
        )
        self.assertEqual(tuple(output["pred_gate_logits"].shape), (2,))
        self.assertEqual(tuple(output["pred_zero_logits"].shape), (2,))
        self.assertEqual(tuple(output["pred_same_event_logits"].shape), (2, 4, 4))
        self.assertTrue(torch.allclose(
            output["pred_same_event_logits"],
            output["pred_same_event_logits"].transpose(1, 2),
        ))

    def test_independent_zero_head_and_loss(self):
        torch.manual_seed(3)
        head = IndependentZeroVerifier(hidden_dim=8, dropout=0.0)
        outputs = {
            "pred_zero_logits": head(
                torch.randn(3, 8),
                torch.randn(3, 4, 2),
                torch.rand(3, 4, 2),
                torch.randn(3, 4),
            )
        }
        loss = independent_zero_loss(
            outputs,
            {"exist_label": torch.tensor([1.0, 0.0, 1.0])},
            positive_query_weight=2.0,
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(head.net[-1].weight.grad.norm()), 0.0)

    def test_pairwise_head_is_symmetric_and_trainable(self):
        torch.manual_seed(7)
        head = PairwiseSameEventHead(8, dropout=0.0, detach_inputs=True)
        logits = head(
            torch.randn(2, 4, 8),
            torch.randn(2, 4, 2),
            torch.tensor([
                [[0.2, 0.2], [0.21, 0.2], [0.7, 0.2], [0.72, 0.2]],
                [[0.2, 0.2], [0.22, 0.2], [0.6, 0.2], [0.62, 0.2]],
            ]),
            torch.randn(2, 6, 8),
            torch.ones(2, 6),
        )
        self.assertEqual(tuple(logits.shape), (2, 4, 4))
        self.assertTrue(torch.allclose(logits, logits.transpose(1, 2)))
        outputs = {
            "pred_same_event_logits": logits,
            "pred_spans": torch.tensor([
                [[0.2, 0.2], [0.21, 0.2], [0.7, 0.2], [0.72, 0.2]],
                [[0.2, 0.2], [0.22, 0.2], [0.6, 0.2], [0.62, 0.2]],
            ]),
        }
        targets = {"span_labels": [
            {"spans": torch.tensor([[0.2, 0.2], [0.7, 0.2]])},
            {"spans": torch.tensor([[0.2, 0.2], [0.6, 0.2]])},
        ]}
        loss = pairwise_same_event_loss(
            outputs, targets, assignment_iou=0.3, ambiguity_margin=0.01
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(head.net[-1].weight.grad.norm()), 0.0)

    def test_two_stage_rescue_and_cautious_veto(self):
        gate = torch.tensor([0.2, 0.8, 0.8, 0.45])
        zero = torch.tensor([0.2, 0.2, 0.9, 0.55])
        localization = torch.tensor([0.7, 0.7, 0.1, 0.3])
        score = combine_two_stage_existence(
            gate, zero, localization,
            mode="cascade", veto_threshold=0.8, localization_threshold=0.2,
        )
        self.assertAlmostEqual(float(score[0]), 0.8)  # stage-two rescue
        self.assertAlmostEqual(float(score[1]), 0.8)
        self.assertAlmostEqual(float(score[2]), 0.1)  # high-zero + weak-local veto
        self.assertAlmostEqual(float(score[3]), 0.45)  # conflict remains permissive
        accepted = two_stage_accept(
            gate, zero, localization,
            gate_threshold=0.3, zero_threshold=0.6,
            veto_threshold=0.8, localization_threshold=0.2,
        )
        self.assertEqual(accepted.tolist(), [True, True, False, True])

    def test_learned_mmr_keeps_fixed_k_but_avoids_duplicate(self):
        scores = torch.tensor([0.9, 0.85, 0.8])
        duplicate = torch.tensor([
            [1.0, 0.95, 0.05],
            [0.95, 1.0, 0.05],
            [0.05, 0.05, 1.0],
        ])
        direct = torch.argsort(scores, descending=True)[:2].tolist()
        learned = learned_mmr_select(
            scores, duplicate, max_output=2, redundancy_lambda=3.0
        ).selected
        self.assertEqual(direct, [0, 1])
        self.assertEqual(learned, [0, 2])
        self.assertEqual(len(direct), len(learned))

    def test_complete_link_fusion_does_not_chain_clusters(self):
        spans = torch.tensor([[0.0, 0.2], [0.01, 0.21], [0.02, 0.22]])
        scores = torch.tensor([0.9, 0.8, 0.7])
        # A~B and B~C, but A!~C.  C must not enter A's cluster.
        duplicate = torch.tensor([
            [1.0, 0.9, 0.2],
            [0.9, 1.0, 0.9],
            [0.2, 0.9, 1.0],
        ])
        fused, fused_scores = cautious_complete_link_fusion(
            spans, scores, duplicate, [0, 2],
            same_event_threshold=0.8, boundary_std_threshold=0.1,
        )
        self.assertEqual(tuple(fused.shape), (2, 2))
        self.assertTrue(torch.allclose(fused_scores, torch.tensor([0.9, 0.7])))
        self.assertTrue(torch.equal(fused[0], spans[0]))
        self.assertGreater(float(fused[1, 0]), float(spans[1, 0]))
        self.assertLess(float(fused[1, 0]), float(spans[2, 0]))


if __name__ == "__main__":
    unittest.main()
