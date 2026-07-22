from __future__ import annotations

import unittest

import torch

from methods.eatr_gmr.build import build_model
from methods.eatr_gmr.checkpoint import (
    config_from_checkpoint,
    detect_state_structure,
    load_parent_state,
)
from methods.eatr_gmr.config import EaTRConfig
from methods.eatr_gmr.criterion import SetCriterion
from methods.eatr_gmr.hierarchical_counter import inverse_sqrt_positive_count_weights
from methods.eatr_gmr.matcher import HungarianEventMatcher, HungarianMatcher
from methods.eatr_gmr.set_decoder import (
    fuse_query_scores,
    hierarchical_two_stage_decode,
)
from methods.eatr_gmr.variants import VARIANT_FLAGS, apply_variant


def tiny_config() -> EaTRConfig:
    return EaTRConfig(
        video_dim=12,
        text_dim=8,
        hidden_dim=16,
        nheads=4,
        enc_layers=1,
        dec_layers=1,
        dim_feedforward=32,
        dropout=0.0,
        input_dropout=0.0,
        num_queries=4,
        num_slot_iter=1,
        n_input_proj=1,
        max_q_l=6,
        max_v_l=8,
        aux_loss=False,
        counter_dropout=0.0,
    )


def model_inputs():
    torch.manual_seed(31)
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


class VariantAndCheckpointTest(unittest.TestCase):
    def test_counter_criterion_receives_long_tail_weights(self):
        config = apply_variant(tiny_config(), "eatr_counter")
        _, criterion = build_model(config)
        expected = inverse_sqrt_positive_count_weights(
            list(config.positive_count_class_counts)
        )
        self.assertTrue(torch.allclose(criterion.positive_count_weights, expected))
        self.assertFalse(torch.allclose(expected, torch.ones_like(expected)))

    def test_checkpoint_structure_detection_overrides_stale_config(self):
        base = tiny_config()
        for variant in VARIANT_FLAGS:
            model, _ = build_model(apply_variant(base, variant))
            structure = detect_state_structure(model.state_dict())
            self.assertEqual(structure["variant"], variant)

            # Simulate an old/stale config: structure must come from state keys.
            detected_config, detected_structure = config_from_checkpoint({
                "config": base.to_dict(),
                "model": model.state_dict(),
            })
            self.assertEqual(detected_structure["variant"], variant)
            expected = VARIANT_FLAGS[variant]
            actual = (
                detected_config.use_exist_head,
                detected_config.use_quality_head,
                detected_config.use_dual_grounding,
                detected_config.use_hierarchical_counter,
            )
            self.assertEqual(actual, expected)

    def test_expanded_variants_preserve_gmr_parent_at_step_zero(self):
        base = tiny_config()
        torch.manual_seed(13)
        parent, _ = build_model(apply_variant(base, "eatr_gmr"))
        parent.eval()
        inputs = model_inputs()
        with torch.no_grad():
            parent_outputs = parent(**inputs)

        for variant in (
            "eatr_quality", "eatr_dual", "eatr_counter", "eatr_hiea2m"
        ):
            child, _ = build_model(apply_variant(base, variant))
            missing = load_parent_state(child, parent.state_dict())
            self.assertTrue(missing)
            child.eval()
            with torch.no_grad():
                child_outputs = child(**inputs)
            for name in ("pred_logits", "pred_spans", "pred_exist_logits"):
                self.assertTrue(
                    torch.equal(parent_outputs[name], child_outputs[name]),
                    msg=f"{variant} changed parent output {name} at step zero",
                )

            if child.quality_embed is not None:
                self.assertEqual(
                    float(child.quality_embed.layers[-1].weight.detach().abs().sum()), 0.0
                )
                quality = child_outputs["pred_quality_logits"]
                fused = fuse_query_scores(
                    parent_outputs["pred_logits"].softmax(-1)[..., 0], quality
                )
                parent_ranking = torch.argsort(
                    parent_outputs["pred_logits"].softmax(-1)[..., 0], dim=-1,
                    descending=True,
                )
                self.assertTrue(torch.equal(parent_ranking, torch.argsort(
                    fused, dim=-1, descending=True
                )))
            if child.dual_grounding is not None:
                self.assertEqual(float(child_outputs["dual_sentence_gate"]), 0.0)
                self.assertEqual(float(child_outputs["dual_phrase_gate"]), 0.0)
            if child.hierarchical_counter is not None:
                self.assertTrue(torch.equal(
                    child_outputs["pred_counter_exist_delta"],
                    torch.zeros_like(child_outputs["pred_counter_exist_delta"]),
                ))


class LossAndDecodeTest(unittest.TestCase):
    def test_strict_gmr_indicator_masks_null_moment_event_and_quality(self):
        criterion = SetCriterion(
            matcher=HungarianMatcher(cost_class=4, cost_span=10, cost_giou=1),
            event_matcher=HungarianEventMatcher(cost_span=10, cost_giou=1),
            weight_dict={
                "loss_span": 10,
                "loss_giou": 1,
                "loss_label": 4,
                "loss_event_span": 30,
                "loss_event_giou": 3,
                "loss_quality": 1,
                "loss_dual_dqa": 0.05,
                "loss_dual_eos": 0.1,
                "loss_span_0": 10,
                "loss_giou_0": 1,
                "loss_label_0": 4,
            },
            aux_loss=True,
            use_quality_head=True,
            use_dual_grounding=True,
            mask_null_vmr_loss=True,
        )
        outputs = {
            "pred_logits": torch.randn(2, 4, 2, requires_grad=True),
            "pred_spans": torch.tensor([
                [[0.2, 0.2], [0.5, 0.25], [0.7, 0.2], [0.8, 0.1]],
                [[0.1, 0.1], [0.3, 0.2], [0.6, 0.2], [0.9, 0.1]],
            ], requires_grad=True),
            "pred_event_spans": torch.tensor([
                [[0.3, 0.2], [0.7, 0.2]],
                [[0.2, 0.1], [0.8, 0.1]],
            ], requires_grad=True),
            # Keep a pseudo event on the null row to prove the explicit mask,
            # rather than relying on an empty feature-derived target.
            "pseudo_event_spans": [
                torch.tensor([[0.5, 0.3]]),
                torch.tensor([[0.4, 0.2]]),
            ],
            "pred_quality_logits": torch.randn(2, 4, requires_grad=True),
            "dual_phrase_attention": torch.rand(2, 3, 6, requires_grad=True),
            "dual_phrase_eos": torch.randn(2, 16, requires_grad=True),
            "dual_text_eos": torch.randn(2, 16),
            "aux_outputs": [{
                "pred_logits": torch.randn(2, 4, 2, requires_grad=True),
                "pred_spans": torch.rand(2, 4, 2, requires_grad=True),
            }],
        }
        targets = {
            "span_labels": [
                {"spans": torch.tensor([[0.5, 0.25]])},
                {"spans": torch.zeros(0, 2)},
            ],
            "exist_label": torch.tensor([1.0, 0.0]),
        }
        losses = criterion(outputs, targets)
        total = criterion.weighted_loss(losses)
        self.assertTrue(torch.isfinite(total))
        total.backward()
        for name in (
            "pred_logits", "pred_spans", "pred_event_spans",
            "pred_quality_logits", "dual_phrase_attention", "dual_phrase_eos",
        ):
            self.assertEqual(
                float(outputs[name].grad[1].abs().sum()), 0.0, msg=name
            )
        for name in ("pred_logits", "pred_spans"):
            self.assertEqual(
                float(outputs["aux_outputs"][0][name].grad[1].abs().sum()),
                0.0, msg=f"aux:{name}",
            )

        all_null_outputs = {
            name: value.detach().clone().requires_grad_(True)
            if isinstance(value, torch.Tensor) else value
            for name, value in outputs.items()
        }
        all_null_outputs["pseudo_event_spans"] = outputs["pseudo_event_spans"]
        all_null_targets = {
            "span_labels": [
                {"spans": torch.zeros(0, 2)},
                {"spans": torch.zeros(0, 2)},
            ],
            "exist_label": torch.zeros(2),
        }
        all_null_losses = criterion(all_null_outputs, all_null_targets)
        for name in (
            "loss_span", "loss_giou", "loss_label", "loss_event_span",
            "loss_event_giou", "loss_quality",
            "loss_dual_dqa", "loss_dual_eos",
        ):
            self.assertEqual(float(all_null_losses[name].detach()), 0.0, msg=name)

    def test_hiea2m_mixed_and_all_null_losses(self):
        model, criterion = build_model(apply_variant(tiny_config(), "eatr_hiea2m"))
        inputs = model_inputs()
        target_sets = {
            "mixed": {
                "span_labels": [
                    {"spans": torch.tensor([[0.5, 0.25]])},
                    {"spans": torch.zeros((0, 2))},
                ],
                "exist_label": torch.tensor([1.0, 0.0]),
                "count_label": torch.tensor([1, 0]),
                "raw_count_label": torch.tensor([1, 0]),
            },
            "all_null": {
                "span_labels": [
                    {"spans": torch.zeros((0, 2))},
                    {"spans": torch.zeros((0, 2))},
                ],
                "exist_label": torch.zeros(2),
                "count_label": torch.zeros(2, dtype=torch.long),
                "raw_count_label": torch.zeros(2, dtype=torch.long),
            },
        }
        for case, targets in target_sets.items():
            outputs = model(**inputs)
            losses = criterion(outputs, targets)
            total = criterion.weighted_loss(losses)
            self.assertTrue(torch.isfinite(total), msg=case)
            if case == "mixed":
                model.zero_grad(set_to_none=True)
                total.backward()
                self.assertGreater(
                    float(model.quality_embed.layers[-1].weight.grad.norm()), 0.0
                )
                self.assertGreater(
                    float(model.dual_grounding.rpg.guides[0][0].weight.grad.norm()), 0.0
                )
                self.assertGreater(
                    float(model.hierarchical_counter.count_head.weight.grad.norm()), 0.0
                )
            if case == "all_null":
                self.assertEqual(float(losses["loss_span"].detach()), 0.0)
                self.assertEqual(float(losses["loss_giou"].detach()), 0.0)
                for name in (
                    "loss_count", "loss_count_ordinal",
                    "loss_count_contrastive", "loss_count_consistency",
                ):
                    self.assertEqual(float(losses[name].detach()), 0.0)

    def test_hierarchical_two_stage_decode(self):
        spans = torch.tensor([[0.0, 0.3], [0.01, 0.31], [0.6, 0.9], [0.4, 0.5]])
        foreground = torch.tensor([0.90, 0.80, 0.70, 0.60])
        quality_logits = torch.tensor([-2.0, 2.0, 0.0, 0.0])
        scores = fuse_query_scores(foreground, quality_logits, quality_alpha=0.5)
        count_two = torch.tensor([0.02, 0.03, 0.90, 0.03, 0.02])

        full = hierarchical_two_stage_decode(
            spans, scores, count_two, mode="full", diversity_lambda=2.0
        )
        adaptive = hierarchical_two_stage_decode(
            spans, scores, count_two, mode="adaptive", diversity_lambda=2.0
        )
        self.assertEqual(full.ranking[0], 1)
        self.assertEqual(full.selected, full.ranking)
        self.assertEqual(adaptive.ranking, full.ranking)
        self.assertEqual(adaptive.selected, full.ranking[:2])
        self.assertEqual(adaptive.predicted_count, 2)

        count_zero = torch.tensor([0.9, 0.05, 0.02, 0.02, 0.01])
        rejected = hierarchical_two_stage_decode(
            spans, scores, count_zero, mode="adaptive", existence_threshold=0.4
        )
        self.assertEqual(rejected.predicted_count, 0)
        self.assertEqual(rejected.selected, [])


if __name__ == "__main__":
    unittest.main()
