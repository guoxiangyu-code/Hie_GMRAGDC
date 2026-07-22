from __future__ import annotations

import json
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from methods.cg_detr_gmr.adapter import GMRExistenceAdapter, existence_loss
from methods.cg_detr_gmr.dataset import SoccerGMRDataset, video_id_to_feature_stem
from methods.cg_detr_gmr.config import VARIANT_FLAGS, detect_variant, finalize_model_arguments
from methods.cg_detr_gmr.engine import build_components
from methods.cg_detr_gmr.evaluate import build_parser as build_eval_parser, merge_checkpoint_args
from methods.cg_detr_gmr.matcher import HungarianMatcher
from methods.cg_detr_gmr.model import SetCriterion
from methods.cg_detr_gmr.smoke import build_parser as build_smoke_parser
from methods.cg_detr_gmr.train import validate_resume_config


def _bare_dataset(clip_length: float = 2.0) -> SoccerGMRDataset:
    dataset = SoccerGMRDataset.__new__(SoccerGMRDataset)
    dataset.clip_length = clip_length
    return dataset


class CGDETRGMRTest(unittest.TestCase):
    def test_resume_rejects_strict_loss_semantic_drift(self) -> None:
        checkpoint = {
            "config": {
                "variant": "cg_detr_gmr",
                "mask_null_vmr_loss": True,
                "epochs": 100,
            }
        }
        matching = SimpleNamespace(
            variant="cg_detr_gmr",
            mask_null_vmr_loss=True,
            epochs=200,
        )
        validate_resume_config(checkpoint, matching)
        matching.mask_null_vmr_loss = False
        with self.assertRaisesRegex(ValueError, "mask_null_vmr_loss"):
            validate_resume_config(checkpoint, matching)

        legacy = {"config": {"variant": "cg_detr_gmr"}}
        validate_resume_config(legacy, matching)
        matching.mask_null_vmr_loss = True
        with self.assertRaisesRegex(ValueError, "mask_null_vmr_loss"):
            validate_resume_config(legacy, matching)

    def test_resume_ignores_runtime_device_resolution(self) -> None:
        checkpoint = {
            "config": {
                "variant": "cg_detr_gmr",
                "device": "cpu",
                "mask_null_vmr_loss": True,
            }
        }
        current = SimpleNamespace(
            variant="cg_detr_gmr",
            device="cuda",
            mask_null_vmr_loss=True,
        )
        validate_resume_config(checkpoint, current)

    def test_evaluation_decoder_controls_are_explicit_runtime_overrides(self) -> None:
        parser = build_eval_parser()
        saved = {
            "decode_mode": "full",
            "existence_threshold": 0.4,
            "quality_alpha": 0.25,
            "diversity_lambda": 0.75,
        }
        implicit = parser.parse_args(["--checkpoint", "model.ckpt"])
        implicit_merged = merge_checkpoint_args(implicit, saved)
        self.assertEqual(implicit_merged["decode_mode"], "full")
        self.assertEqual(implicit_merged["quality_alpha"], 0.25)
        self.assertEqual(implicit_merged["diversity_lambda"], 0.75)

        explicit = parser.parse_args([
            "--checkpoint", "model.ckpt",
            "--decode_mode", "adaptive",
            "--existence_threshold", "0.37",
            "--quality_alpha", "0.4",
            "--diversity_lambda", "1.5",
        ])
        explicit_merged = merge_checkpoint_args(explicit, saved)
        self.assertEqual(explicit_merged["decode_mode"], "adaptive")
        self.assertEqual(explicit_merged["existence_threshold"], 0.37)
        self.assertEqual(explicit_merged["quality_alpha"], 0.4)
        self.assertEqual(explicit_merged["diversity_lambda"], 1.5)

    def test_checkpoint_structure_detection(self) -> None:
        prefixes = {
            "exist": "exist_head.0.weight",
            "quality": "quality_embed.layers.0.weight",
            "phrase": "phrase_grounding.phrase_gate_logit",
            "counter": "hierarchical_counter.exist_head.weight",
        }
        for variant, flags in VARIANT_FLAGS.items():
            state = {
                prefixes[name]: torch.zeros(1)
                for name, enabled in zip(
                    ("exist", "quality", "phrase", "counter"), flags
                ) if enabled
            }
            state["class_embed.weight"] = torch.zeros(1)
            detected, _ = detect_variant(state)
            self.assertEqual(detected, variant)

    def test_hiea2m_preserves_gmr_parent_at_step_zero(self) -> None:
        def model_args(variant: str):
            args = build_smoke_parser().parse_args(["--device", "cpu"])
            args.variant = variant
            return finalize_model_arguments(args)

        torch.manual_seed(2)
        parent, _, _ = build_components(model_args("cg_detr_gmr"))
        child, _, _ = build_components(model_args("cg_hiea2m"))
        incompatible = child.load_state_dict(parent.state_dict(), strict=False)
        self.assertFalse(incompatible.unexpected_keys)
        self.assertTrue(incompatible.missing_keys)
        self.assertTrue(all(name.startswith((
            "quality_embed.", "phrase_grounding.", "hierarchical_counter.",
        )) for name in incompatible.missing_keys))

        inputs = {
            "src_vid": torch.randn(1, 7, 2818),
            "src_vid_mask": torch.tensor([[1, 1, 1, 1, 1, 0, 0.0]]),
            "src_txt": torch.randn(1, 6, 512),
            "src_txt_mask": torch.tensor([[1, 1, 1, 1, 0, 0.0]]),
        }
        parent.eval()
        child.eval()
        with torch.no_grad():
            parent_outputs = parent(**inputs)
            child_outputs = child(**inputs)
        for name in ("pred_logits", "pred_spans", "pred_exist_logits"):
            self.assertTrue(torch.equal(parent_outputs[name], child_outputs[name]))
        self.assertEqual(float(child_outputs["dual_phrase_gate"]), 0.0)
        self.assertTrue(torch.equal(
            child_outputs["pred_counter_exist_delta"],
            torch.zeros_like(child_outputs["pred_counter_exist_delta"]),
        ))

    def test_extension_free_feature_stem(self) -> None:
        self.assertEqual(video_id_to_feature_stem("match.mp4"), "match")
        self.assertEqual(video_id_to_feature_stem("match"), "match")

    def test_relevant_clips_union_uses_every_window(self) -> None:
        dataset = _bare_dataset()
        mask = dataset._relevant_clips([[1.0, 3.0], [6.0, 8.0]], context_length=5)
        self.assertTrue(torch.equal(mask, torch.tensor([1.0, 1.0, 0.0, 1.0, 0.0])))

    def test_relevant_clips_null_is_all_zero(self) -> None:
        dataset = _bare_dataset()
        self.assertTrue(torch.equal(dataset._relevant_clips([], 4), torch.zeros(4)))

    def test_existence_adapter_binary_loss(self) -> None:
        adapter = GMRExistenceAdapter(16)
        logits = adapter(torch.randn(3, 10, 16))
        loss = existence_loss(logits, torch.tensor([1.0, 0.0, 1.0]))
        self.assertEqual(logits.shape, (3,))
        self.assertTrue(torch.isfinite(loss))

    def test_all_null_detr_loss_is_finite(self) -> None:
        matcher = HungarianMatcher(cost_class=4, cost_span=10, cost_giou=1)
        criterion = SetCriterion(
            matcher=matcher,
            weight_dict={"loss_span": 10, "loss_giou": 1, "loss_label": 4},
            eos_coef=0.1,
            losses=["spans", "labels"],
            temperature=0.07,
            span_loss_type="l1",
            max_v_l=75,
            args=SimpleNamespace(device="cpu"),
        )
        outputs = {
            "pred_logits": torch.randn(2, 10, 2, requires_grad=True),
            "pred_spans": torch.rand(2, 10, 2, requires_grad=True),
        }
        targets = {
            "span_labels": [
                {"spans": torch.zeros(0, 2)},
                {"spans": torch.zeros(0, 2)},
            ]
        }
        losses = criterion(outputs, targets)
        total = sum(losses[key] * criterion.weight_dict[key] for key in criterion.weight_dict)
        self.assertTrue(torch.isfinite(total))
        total.backward()
        self.assertIsNotNone(outputs["pred_logits"].grad)

    def test_strict_gmr_indicator_removes_null_vmr_gradients(self) -> None:
        matcher = HungarianMatcher(cost_class=4, cost_span=10, cost_giou=1)
        criterion = SetCriterion(
            matcher=matcher,
            weight_dict={"loss_span": 10, "loss_giou": 1, "loss_label": 4},
            eos_coef=0.1,
            losses=["spans", "labels"],
            temperature=0.07,
            span_loss_type="l1",
            max_v_l=75,
            args=SimpleNamespace(device="cpu", mask_null_vmr_loss=True),
        )
        torch.manual_seed(103)
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
            losses[key] * criterion.weight_dict[key]
            for key in criterion.weight_dict
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

    def test_strict_mixed_matches_positive_only_with_quality_and_aux(self) -> None:
        matcher = HungarianMatcher(cost_class=4, cost_span=10, cost_giou=1)
        weights = {
            "loss_span": 10, "loss_giou": 1, "loss_label": 4,
            "loss_quality": 1, "loss_contrastive_align": 1,
            "loss_span_0": 10, "loss_giou_0": 1,
            "loss_label_0": 4, "loss_contrastive_align_0": 1,
        }
        criterion = SetCriterion(
            matcher=matcher, weight_dict=weights, eos_coef=0.1,
            losses=["spans", "labels", "quality", "contrastive_align"],
            temperature=0.07, span_loss_type="l1", max_v_l=75,
            args=SimpleNamespace(device="cpu", mask_null_vmr_loss=True),
        )
        torch.manual_seed(113)

        def layer_outputs():
            return {
                "pred_logits": torch.randn(2, 4, 2, requires_grad=True),
                "pred_spans": torch.rand(2, 4, 2, requires_grad=True),
                "proj_queries": torch.randn(2, 4, 5, requires_grad=True),
                "proj_txt_mem": torch.randn(2, 3, 5, requires_grad=True),
            }

        outputs = layer_outputs()
        outputs["pred_quality_logits"] = torch.randn(2, 4, requires_grad=True)
        outputs["aux_outputs"] = [layer_outputs()]
        targets = {
            "span_labels": [
                {"spans": torch.tensor([[0.5, 0.25]])},
                {"spans": torch.zeros(0, 2)},
            ],
            "exist_label": torch.tensor([1.0, 0.0]),
        }
        positive_outputs = {
            name: value[:1]
            for name, value in outputs.items()
            if isinstance(value, torch.Tensor)
        }
        positive_outputs["aux_outputs"] = [{
            name: value[:1]
            for name, value in outputs["aux_outputs"][0].items()
        }]
        positive_targets = {
            "span_labels": targets["span_labels"][:1],
            "exist_label": targets["exist_label"][:1],
        }
        mixed_losses = criterion(outputs, targets)
        positive_losses = criterion(positive_outputs, positive_targets)
        for name in weights:
            self.assertTrue(
                torch.allclose(mixed_losses[name], positive_losses[name]), msg=name
            )

        sum(mixed_losses[name] * weight for name, weight in weights.items()).backward()
        for name in (
            "pred_logits", "pred_spans", "pred_quality_logits",
            "proj_queries", "proj_txt_mem",
        ):
            self.assertEqual(float(outputs[name].grad[1].abs().sum()), 0.0, msg=name)
        for name in ("pred_logits", "pred_spans", "proj_queries", "proj_txt_mem"):
            self.assertEqual(
                float(outputs["aux_outputs"][0][name].grad[1].abs().sum()),
                0.0, msg=f"aux:{name}",
            )

    def test_source_manifest_pins_revision(self) -> None:
        path = Path(__file__).parents[1] / "methods/cg_detr_gmr/SOURCE.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(
            manifest["upstream_revision"],
            "212b2e49a0893512930d63b9294a8c066347e606",
        )
        self.assertEqual(manifest["license"], "MIT")
