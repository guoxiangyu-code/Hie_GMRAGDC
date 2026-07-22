from __future__ import annotations

import unittest

import torch

from eval.eval_main import compute_count_diagnostics
from methods.eatr_gmr.hierarchical_counter import (
    hierarchical_counter_losses as eatr_hierarchical_counter_losses,
)
from models.moment_detr_gmr.dual_grounding import (
    TemporalDualGrounding,
    dual_grounding_losses,
)
from models.moment_detr_gmr.hierarchical_counter import (
    HierarchicalMomentCounter,
    hierarchical_count_probabilities,
    hierarchical_counter_losses,
)
from models.moment_detr_gmr.matcher import HungarianMatcher
from models.moment_detr_gmr.moment_detr import SetCriterion
from models.moment_detr_gmr.set_decoder import (
    adaptive_count_indices,
    diversity_ranking,
    fuse_query_scores,
)


class DualGroundingTest(unittest.TestCase):
    def test_shapes_losses_and_gradient(self):
        torch.manual_seed(4)
        module = TemporalDualGrounding(
            hidden_dim=16,
            nheads=4,
            dropout=0.0,
            num_phrases=3,
            num_dummies=3,
            max_text_len=8,
            phrase_slot_iterations=1,
        )
        video = torch.randn(2, 5, 16, requires_grad=True)
        text = torch.randn(2, 8, 16, requires_grad=True)
        video_mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])
        text_mask = torch.tensor([
            [1, 1, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0, 0],
        ])
        result = module(video, video_mask, text, text_mask)
        self.assertEqual(result.video.shape, video.shape)
        self.assertEqual(result.phrase_attention.shape, (2, 3, 8))
        self.assertEqual(result.phrase_eos.shape, (2, 16))
        self.assertTrue(torch.all(result.video[1, 3:] == 0))
        self.assertEqual(float(result.sentence_gate.detach()), 0.0)
        self.assertEqual(float(result.phrase_gate.detach()), 0.0)

        changed_padding = text.detach().clone()
        changed_padding[0, 5:] = 1000
        changed_padding[1, 4:] = -1000
        repeated = module(video.detach(), video_mask, changed_padding, text_mask)
        self.assertTrue(torch.equal(result.phrase_eos, repeated.phrase_eos))

        outputs = {
            "dual_phrase_attention": result.phrase_attention,
            "dual_phrase_eos": result.phrase_eos,
            "dual_text_eos": result.text_eos,
        }
        losses = dual_grounding_losses(outputs)
        total = result.video.square().mean() + sum(losses.values())
        total.backward()
        self.assertTrue(torch.isfinite(total))
        self.assertIsNotNone(module.rpg.guides[0][0].weight.grad)
        self.assertGreater(float(module.rpg.guides[0][0].weight.grad.norm()), 0.0)
        self.assertIsNotNone(module.phrase_gate_logit.grad)

    def test_sample_mask_excludes_null_rows_from_dual_losses(self):
        attention = torch.rand(2, 3, 6, requires_grad=True)
        phrase_eos = torch.randn(2, 16, requires_grad=True)
        text_eos = torch.randn(2, 16)
        outputs = {
            "dual_phrase_attention": attention,
            "dual_phrase_eos": phrase_eos,
            "dual_text_eos": text_eos,
        }
        losses = dual_grounding_losses(
            outputs, sample_mask=torch.tensor([True, False])
        )
        sum(losses.values()).backward()
        self.assertEqual(float(attention.grad[1].abs().sum()), 0.0)
        self.assertEqual(float(phrase_eos.grad[1].abs().sum()), 0.0)

        all_null = dual_grounding_losses(
            outputs, sample_mask=torch.tensor([False, False])
        )
        self.assertEqual(float(all_null["loss_dual_dqa"].detach()), 0.0)
        self.assertEqual(float(all_null["loss_dual_eos"].detach()), 0.0)


class StrictMomentGMRLossTest(unittest.TestCase):
    def test_null_rows_have_no_original_vmr_gradient(self):
        criterion = SetCriterion(
            matcher=HungarianMatcher(
                cost_class=4, cost_span=10, cost_giou=1,
                span_loss_type="l1",
            ),
            weight_dict={"loss_span": 10, "loss_giou": 1, "loss_label": 4},
            eos_coef=0.1,
            losses=["spans", "labels"],
            span_loss_type="l1",
            max_v_l=75,
            mask_null_vmr_loss=True,
        )
        torch.manual_seed(107)
        outputs = {
            "pred_logits": torch.randn(2, 4, 2, requires_grad=True),
            "pred_spans": torch.rand(2, 4, 2, requires_grad=True),
        }
        targets = {
            "span_labels": [
                {"spans": torch.tensor([[0.5, 0.25]])},
                {"spans": torch.zeros(0, 2)},
            ],
            "exist_label": torch.tensor([1.0, 0.0]),
        }
        losses = criterion(outputs, targets)
        total = sum(
            losses[name] * weight
            for name, weight in criterion.weight_dict.items()
        )
        self.assertTrue(torch.isfinite(total))
        total.backward()
        self.assertEqual(float(outputs["pred_logits"].grad[1].abs().sum()), 0.0)
        self.assertEqual(float(outputs["pred_spans"].grad[1].abs().sum()), 0.0)

        all_null_outputs = {
            "pred_logits": torch.randn(2, 4, 2, requires_grad=True),
            "pred_spans": torch.rand(2, 4, 2, requires_grad=True),
        }
        all_null_targets = {
            "span_labels": [
                {"spans": torch.zeros(0, 2)},
                {"spans": torch.zeros(0, 2)},
            ],
            "exist_label": torch.zeros(2),
        }
        all_null_losses = criterion(all_null_outputs, all_null_targets)
        for name in ("loss_span", "loss_giou", "loss_label"):
            self.assertEqual(float(all_null_losses[name].detach()), 0.0)


class HierarchicalCounterTest(unittest.TestCase):
    def _outputs(self):
        torch.manual_seed(7)
        module = HierarchicalMomentCounter(hidden_dim=16, dropout=0.0)
        outputs = module(
            decoder_queries=torch.randn(6, 10, 16),
            pred_logits=torch.randn(6, 10, 2),
            text_memory=torch.randn(6, 7, 16),
            text_mask=torch.ones(6, 7),
            video_memory=torch.randn(6, 5, 16),
            video_mask=torch.ones(6, 5),
        )
        return module, outputs

    def test_factorized_probabilities_and_losses(self):
        module, outputs = self._outputs()
        probabilities = hierarchical_count_probabilities(outputs)
        self.assertEqual(probabilities.shape, (6, 5))
        self.assertTrue(torch.allclose(probabilities.sum(dim=1), torch.ones(6), atol=1e-6))
        targets = {
            "exist_label": torch.tensor([0, 1, 1, 1, 1, 0], dtype=torch.float32),
            "count_label": torch.tensor([0, 1, 2, 3, 4, 0]),
        }
        losses = hierarchical_counter_losses(outputs, targets)
        total = sum(losses.values())
        total.backward()
        self.assertTrue(all(torch.isfinite(value) for value in losses.values()))
        self.assertGreater(float(module.exist_head.weight.grad.norm()), 0.0)
        self.assertGreater(float(module.count_head.weight.grad.norm()), 0.0)

    def test_all_null_batch_has_zero_conditional_losses(self):
        _, outputs = self._outputs()
        targets = {
            "exist_label": torch.zeros(6),
            "count_label": torch.zeros(6, dtype=torch.long),
        }
        losses = hierarchical_counter_losses(outputs, targets)
        for name in [
            "loss_count",
            "loss_count_ordinal",
            "loss_count_contrastive",
            "loss_count_consistency",
        ]:
            self.assertEqual(float(losses[name]), 0.0)

    def test_open_ended_count_does_not_force_soft_count_to_four(self):
        _, outputs = self._outputs()
        targets = {
            "exist_label": torch.ones(6),
            "count_label": torch.full((6,), 4, dtype=torch.long),
            "raw_count_label": torch.full((6,), 5, dtype=torch.long),
        }
        losses = hierarchical_counter_losses(outputs, targets)
        self.assertEqual(float(losses["loss_count_consistency"].detach()), 0.0)

    def test_existence_loss_inherits_relative_multi_count_weights(self):
        _, outputs = self._outputs()
        outputs["pred_exist_logits"] = torch.tensor(
            [1.5, -0.5, -1.0, -1.5, -2.0, 0.5], requires_grad=True
        )
        targets = {
            "exist_label": torch.tensor([0, 1, 1, 1, 1, 0], dtype=torch.float32),
            "count_label": torch.tensor([0, 1, 2, 3, 4, 0]),
        }
        class_weights = torch.tensor([1.0, 2.0, 4.0, 8.0])
        raw = torch.nn.functional.binary_cross_entropy_with_logits(
            outputs["pred_exist_logits"], targets["exist_label"], reduction="none"
        )
        sample_weights = torch.tensor([1.0, 1.0, 2.0, 4.0, 8.0, 1.0])
        expected = (raw * sample_weights).sum() / sample_weights.sum()
        for loss_function in (
            hierarchical_counter_losses, eatr_hierarchical_counter_losses
        ):
            losses = loss_function(
                outputs, targets, positive_count_weights=class_weights
            )
            self.assertTrue(torch.allclose(losses["loss_exist"], expected))


class SetDecoderTest(unittest.TestCase):
    def test_quality_fusion_and_diversity(self):
        foreground = torch.tensor([0.9, 0.8, 0.7])
        quality_logits = torch.tensor([-2.0, 2.0, 0.0])
        fused = fuse_query_scores(foreground, quality_logits, quality_alpha=0.5)
        self.assertGreater(float(fused[1]), float(fused[0]))

        spans = torch.tensor([[0.0, 0.4], [0.02, 0.42], [0.6, 0.8]])
        ranking = diversity_ranking(spans, torch.tensor([0.9, 0.89, 0.7]), diversity_lambda=2.0)
        self.assertEqual(ranking[0], 0)
        self.assertEqual(ranking[1], 2)

    def test_adaptive_count(self):
        ranking = [3, 2, 1, 0]
        scores = torch.tensor([0.1, 0.2, 0.8, 0.9])
        count_two = torch.tensor([0.02, 0.03, 0.90, 0.03, 0.02])
        self.assertEqual(
            adaptive_count_indices(ranking, scores, count_two, mode="adaptive"),
            [3, 2],
        )
        count_zero = torch.tensor([0.9, 0.05, 0.02, 0.02, 0.01])
        self.assertEqual(
            adaptive_count_indices(ranking, scores, count_zero, mode="hard"),
            [],
        )
        uncertain_positive = torch.tensor([0.4, 0.15, 0.15, 0.15, 0.15])
        self.assertEqual(
            adaptive_count_indices(
                ranking,
                scores,
                uncertain_positive,
                mode="hard",
                existence_threshold=0.4,
            ),
            [3],
        )
        self.assertEqual(
            adaptive_count_indices(
                ranking,
                scores,
                None,
                mode="threshold",
                window_score_threshold=0.5,
            ),
            [3, 2],
        )


class CountDiagnosticsTest(unittest.TestCase):
    def test_count_metrics(self):
        ground_truth = [
            {"qid": 1, "relevant_windows": []},
            {"qid": 2, "relevant_windows": [[0, 1], [2, 3]]},
        ]
        submission = [
            {"qid": 1, "pred_count": 0, "pred_count_probs": [1, 0, 0, 0, 0]},
            {"qid": 2, "pred_count": 1, "pred_count_probs": [0, 1, 0, 0, 0]},
        ]
        result = compute_count_diagnostics(submission, ground_truth)
        self.assertEqual(result["Count-Acc"], 50.0)
        self.assertEqual(result["Positive-Count-Acc"], 0.0)
        self.assertEqual(result["support"], [1, 0, 1, 0, 0])


if __name__ == "__main__":
    unittest.main()
