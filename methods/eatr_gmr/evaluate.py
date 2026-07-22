"""Evaluate an EaTR/EaTR-GMR checkpoint with the official GMR evaluator.

Run as ``python -m methods.eatr_gmr.evaluate --help`` from the repository root.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .build import build_model
from .checkpoint import config_from_checkpoint
from .cli import add_data_arguments, add_runtime_arguments
from .dataset import SoccerGMRDataset, collate_fn
from .runtime import official_metrics, predict_views, resolve_device, write_json, write_jsonl


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate EaTR/EaTR-GMR on Soccer-GMR",
        allow_abbrev=False,
    )
    add_data_arguments(parser, require_train=False)
    add_runtime_arguments(parser)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--no-metrics", action="store_true", help="prediction-only mode for unlabeled test data")
    parser.add_argument("--max-predictions", type=int, default=10)
    parser.add_argument("--no-round-to-clip", action="store_true")
    parser.add_argument("--quality-score-alpha", type=float, default=None)
    parser.add_argument("--diversity-lambda", type=float, default=None)
    parser.add_argument("--count-exist-threshold", type=float, default=None)
    parser.add_argument("--count-confidence-threshold", type=float, default=None)
    parser.add_argument("--window-score-threshold", type=float, default=None)
    parser.add_argument("--diagnostic-mode", choices=("adaptive", "hard"), default="adaptive")
    parser.add_argument("--no-adaptive-diagnostics", action="store_true")
    parser.add_argument("--save-raw-queries", action="store_true")
    return parser


def main(argv=None) -> None:
    args = make_parser().parse_args(argv)
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config, structure = config_from_checkpoint(checkpoint)
    if config.video_dim != args.video_feature_dim + 2:
        raise ValueError("checkpoint and --video-feature-dim disagree")
    if config.text_dim != args.text_feature_dim:
        raise ValueError("checkpoint and --text-feature-dim disagree")

    model, _ = build_model(config)
    model.load_state_dict(checkpoint.get("model", checkpoint), strict=True)
    model.to(device)
    dataset = SoccerGMRDataset(
        annotation_path=args.annotations,
        slowfast_dir=args.slowfast_dir,
        clip_dir=args.clip_dir,
        text_dir=args.text_dir,
        max_video_len=args.max_video_len,
        max_text_len=args.max_text_len,
        clip_length=args.clip_length,
        max_windows=args.max_windows,
        load_labels=not args.no_metrics,
        trim_text_by_attention_mask=args.trim_text_by_attention_mask,
        expected_video_feature_dim=args.video_feature_dim,
        expected_text_feature_dim=args.text_feature_dim,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )
    include_diagnostic = (
        config.use_hierarchical_counter and not args.no_adaptive_diagnostics
    )
    modes = ("full", args.diagnostic_mode) if include_diagnostic else ("full",)
    views = predict_views(
        model, loader, device, modes=modes, max_predictions=args.max_predictions,
        round_to_clip=not args.no_round_to_clip, clip_length=args.clip_length,
        quality_alpha=(
            config.quality_score_alpha
            if args.quality_score_alpha is None else args.quality_score_alpha
        ),
        diversity_lambda=(
            config.diversity_lambda
            if args.diversity_lambda is None else args.diversity_lambda
        ),
        existence_threshold=(
            config.count_exist_threshold
            if args.count_exist_threshold is None else args.count_exist_threshold
        ),
        count_confidence_threshold=(
            config.count_confidence_threshold
            if args.count_confidence_threshold is None
            else args.count_confidence_threshold
        ),
        window_score_threshold=(
            config.window_score_threshold
            if args.window_score_threshold is None else args.window_score_threshold
        ),
        save_raw_queries=args.save_raw_queries,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    submission = views["full"]
    submission_path = output_dir / "submission.jsonl"
    write_jsonl(submission, submission_path)
    result = {
        "submission": str(submission_path),
        "num_predictions": len(submission),
        "detected_structure": structure,
        "primary_decode_mode": "full",
    }
    if not args.no_metrics:
        metrics = official_metrics(submission, dataset.data, map_num_workers=1, verbose=True)
        metrics_path = output_dir / "metrics.json"
        write_json(metrics, metrics_path)
        result["metrics"] = str(metrics_path)
        result["brief"] = metrics["brief"]
    if include_diagnostic:
        diagnostic = views[args.diagnostic_mode]
        diagnostic_path = output_dir / f"submission_{args.diagnostic_mode}.jsonl"
        write_jsonl(diagnostic, diagnostic_path)
        result[f"{args.diagnostic_mode}_submission"] = str(diagnostic_path)
        if not args.no_metrics:
            diagnostic_metrics = official_metrics(
                diagnostic, dataset.data, map_num_workers=1, verbose=False
            )
            diagnostic_metrics_path = output_dir / f"metrics_{args.diagnostic_mode}.json"
            write_json(diagnostic_metrics, diagnostic_metrics_path)
            result[f"{args.diagnostic_mode}_metrics"] = str(diagnostic_metrics_path)
            result[f"{args.diagnostic_mode}_brief"] = diagnostic_metrics["brief"]
    write_json(result, output_dir / "evaluation_manifest.json")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
