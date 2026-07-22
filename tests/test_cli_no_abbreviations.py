"""Regression tests for exact option spelling in preregistered CLI steps."""

from __future__ import annotations

import contextlib
import io
import unittest

from eval.eval_main import build_parser as build_eval_parser
from methods.cg_detr_gmr.evaluate import build_parser as build_cg_parser
from methods.eatr_gmr.evaluate import make_parser as build_eatr_parser
from methods.qd_detr_gmr.evaluate import build_parser as build_qd_parser
from scripts.diagnose_gmr_groups import build_parser as build_diagnose_parser
from scripts.fuse_gmr_heads import build_parser as build_fuse_parser
from training.moment_detr_gmr.evaluate import build_parser as build_moment_parser


class ExactCLIOptionTest(unittest.TestCase):
    def assert_rejected(self, parser, argv: list[str]) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                parser.parse_args(argv)
        self.assertEqual(raised.exception.code, 2)

    def assert_exact_parser(self, parser) -> None:
        self.assertFalse(parser.allow_abbrev)

    def test_moment_evaluate_requires_exact_options(self) -> None:
        parser = build_moment_parser()
        self.assert_exact_parser(parser)
        canonical = [
            "--model_path", "model.ckpt", "--split", "val",
            "--eval_path", "val.jsonl",
        ]
        parsed = parser.parse_args(canonical)
        self.assertEqual(parsed.model_path, "model.ckpt")
        self.assert_rejected(parser, [
            "--model_p", "model.ckpt", "--split", "val",
            "--eval_path", "val.jsonl",
        ])

    def test_qd_evaluate_requires_exact_options(self) -> None:
        parser = build_qd_parser()
        self.assert_exact_parser(parser)
        parsed = parser.parse_args([
            "--checkpoint", "model.ckpt", "--eval_annotation", "val.jsonl",
        ])
        self.assertEqual(parsed.checkpoint, "model.ckpt")
        self.assertEqual(parsed.eval_annotation, "val.jsonl")
        self.assert_rejected(parser, ["--checkp", "model.ckpt"])
        self.assert_rejected(parser, [
            "--checkpoint", "model.ckpt", "--eval_annot", "val.jsonl",
        ])

    def test_cg_evaluate_requires_exact_options(self) -> None:
        parser = build_cg_parser()
        self.assert_exact_parser(parser)
        parsed = parser.parse_args([
            "--checkpoint", "model.ckpt", "--eval_annotation", "val.jsonl",
        ])
        self.assertEqual(parsed.checkpoint, "model.ckpt")
        self.assertEqual(parsed.eval_annotation, "val.jsonl")
        self.assert_rejected(parser, ["--checkp", "model.ckpt"])
        self.assert_rejected(parser, [
            "--checkpoint", "model.ckpt", "--eval_annot", "val.jsonl",
        ])

    def test_eatr_evaluate_requires_exact_options(self) -> None:
        parser = build_eatr_parser()
        self.assert_exact_parser(parser)
        canonical = [
            "--annotations", "val.jsonl",
            "--slowfast-dir", "slowfast",
            "--clip-dir", "clip",
            "--text-dir", "text",
            "--checkpoint", "model.ckpt",
            "--output-dir", "out",
        ]
        parsed = parser.parse_args(canonical)
        self.assertEqual(parsed.checkpoint, "model.ckpt")
        abbreviated = canonical.copy()
        abbreviated[8] = "--checkp"
        self.assert_rejected(parser, abbreviated)

    def test_fusion_requires_exact_options(self) -> None:
        parser = build_fuse_parser()
        self.assert_exact_parser(parser)
        canonical = [
            "--localization", "loc.jsonl",
            "--decision", "decision.jsonl",
            "--output", "fused.jsonl",
            "--ground-truth", "gt.jsonl",
        ]
        parsed = parser.parse_args(canonical)
        self.assertEqual(parsed.ground_truth, "gt.jsonl")
        abbreviated = canonical.copy()
        abbreviated[6] = "--ground-t"
        self.assert_rejected(parser, abbreviated)

    def test_official_eval_requires_exact_options(self) -> None:
        parser = build_eval_parser()
        self.assert_exact_parser(parser)
        canonical = [
            "--submission_path", "submission.jsonl",
            "--gt_path", "gt.jsonl",
            "--save_path", "metrics.json",
        ]
        parsed = parser.parse_args(canonical)
        self.assertEqual(parsed.submission_path, "submission.jsonl")
        abbreviated = canonical.copy()
        abbreviated[0] = "--submiss"
        self.assert_rejected(parser, abbreviated)

    def test_diagnostics_require_exact_options(self) -> None:
        parser = build_diagnose_parser()
        self.assert_exact_parser(parser)
        canonical = [
            "--submission", "submission.jsonl",
            "--ground-truth", "gt.jsonl",
            "--output", "report.json",
        ]
        parsed = parser.parse_args(canonical)
        self.assertEqual(parsed.ground_truth, "gt.jsonl")
        abbreviated = canonical.copy()
        abbreviated[2] = "--ground-t"
        self.assert_rejected(parser, abbreviated)


if __name__ == "__main__":
    unittest.main()
