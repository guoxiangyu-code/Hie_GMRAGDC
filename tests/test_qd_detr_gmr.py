from __future__ import annotations

import json
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from methods.qd_detr_gmr.adapter import GMRExistenceAdapter, existence_loss
from methods.qd_detr_gmr.dataset import video_id_to_feature_stem
from methods.qd_detr_gmr.matcher import HungarianMatcher
from methods.qd_detr_gmr.model import SetCriterion
from methods.qd_detr_gmr.train import validate_resume_config


class QDDETRGMRTest(unittest.TestCase):
    def test_resume_rejects_strict_loss_semantic_drift(self):
        checkpoint = {
            "config": {
                "variant": "qd_detr_gmr",
                "mask_null_vmr_loss": True,
                "epochs": 100,
            }
        }
        matching = SimpleNamespace(
            variant="qd_detr_gmr",
            mask_null_vmr_loss=True,
            epochs=200,
        )
        validate_resume_config(checkpoint, matching)
        matching.mask_null_vmr_loss = False
        with self.assertRaisesRegex(ValueError, "mask_null_vmr_loss"):
            validate_resume_config(checkpoint, matching)

        legacy = {"config": {"variant": "qd_detr_gmr"}}
        validate_resume_config(legacy, matching)
        matching.mask_null_vmr_loss = True
        with self.assertRaisesRegex(ValueError, "mask_null_vmr_loss"):
            validate_resume_config(legacy, matching)

    def test_resume_ignores_runtime_device_resolution(self):
        checkpoint = {
            "config": {
                "variant": "qd_detr_gmr",
                "device": "cpu",
                "mask_null_vmr_loss": True,
            }
        }
        current = SimpleNamespace(
            variant="qd_detr_gmr",
            device="cuda",
            mask_null_vmr_loss=True,
        )
        validate_resume_config(checkpoint, current)

    def test_video_extension_is_removed_for_feature_lookup(self):
        self.assertEqual(video_id_to_feature_stem("match.clip.mp4"), "match.clip")
        self.assertEqual(video_id_to_feature_stem("already_a_stem"), "already_a_stem")

    def test_existence_adapter_has_finite_binary_loss(self):
        adapter = GMRExistenceAdapter(16)
        logits = adapter(torch.randn(3, 10, 16))
        loss = existence_loss(logits, torch.tensor([1.0, 0.0, 1.0]))
        self.assertEqual(logits.shape, (3,))
        self.assertTrue(torch.isfinite(loss))

    def test_all_null_batch_has_finite_detr_losses(self):
        matcher = HungarianMatcher(cost_class=4, cost_span=10, cost_giou=1)
        criterion = SetCriterion(
            matcher=matcher,
            weight_dict={"loss_span": 10, "loss_giou": 1, "loss_label": 4},
            eos_coef=0.1,
            losses=["spans", "labels"],
            temperature=0.07,
            span_loss_type="l1",
            max_v_l=75,
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

    def test_strict_gmr_indicator_removes_null_vmr_gradients(self):
        matcher = HungarianMatcher(cost_class=4, cost_span=10, cost_giou=1)
        criterion = SetCriterion(
            matcher=matcher,
            weight_dict={"loss_span": 10, "loss_giou": 1, "loss_label": 4},
            eos_coef=0.1,
            losses=["spans", "labels"],
            temperature=0.07,
            span_loss_type="l1",
            max_v_l=75,
            mask_null_vmr_loss=True,
        )
        torch.manual_seed(101)
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

    def test_strict_mixed_matches_positive_only_with_quality_and_aux(self):
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
            mask_null_vmr_loss=True,
        )
        torch.manual_seed(109)

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

    def test_source_manifest_pins_requested_revision(self):
        path = Path(__file__).parents[1] / "methods/qd_detr_gmr/SOURCE.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["revision"], "f8628f79f7c651b586300b142dbe9b85e43857cc")
        self.assertEqual(manifest["license"], "MIT")
