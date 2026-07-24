"""Evaluate a QD-DETR(-GMR) checkpoint with the official Soccer-GMR metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .config import REPO_ROOT, add_data_arguments, add_model_arguments, finalize_model_arguments
from .engine import (
    build_components,
    evaluate_submission,
    make_dataset,
    make_loader,
    predict_modes,
    save_json,
    save_jsonl,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    add_data_arguments(parser)
    add_model_arguments(parser)
    parser.set_defaults(eval_annotation=str(REPO_ROOT / "data/label/Standard/test.jsonl"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--submission_path", default=None)
    parser.add_argument("--metrics_path", default=None)
    parser.add_argument("--gmiou_threshold", type=float, default=0.4)
    parser.add_argument("--map_num_workers", type=int, default=8)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--round_to_clip", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow_partial_load", action="store_true")
    parser.add_argument("--diagnostic_decoders", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save_raw_queries", action="store_true")
    return parser


def detect_variant(state: dict[str, torch.Tensor]) -> tuple[str, dict[str, bool]]:
    keys = tuple(state.keys())
    structure = {
        "exist_head": any(key.startswith("exist_head.") for key in keys),
        "quality_head": any(key.startswith("quality_embed.") for key in keys),
        "dual_grounding": any(key.startswith("dual_grounding.") for key in keys),
        "hierarchical_counter": any(key.startswith("hierarchical_counter.") for key in keys),
    }
    if structure["quality_head"] and structure["dual_grounding"] and structure["hierarchical_counter"]:
        variant = "qd_hiea2m"
    elif structure["quality_head"] and structure["dual_grounding"]:
        variant = "qd_quality_dual"
    elif structure["hierarchical_counter"]:
        variant = "qd_counter"
    elif structure["dual_grounding"]:
        variant = "qd_dual"
    elif structure["quality_head"]:
        variant = "qd_quality"
    elif structure["exist_head"]:
        variant = "qd_detr_gmr"
    else:
        variant = "qd_detr"
    return variant, structure


def main(argv: list[str] | None = None) -> None:
    cli = build_parser().parse_args(argv)
    checkpoint = torch.load(cli.checkpoint, map_location="cpu", weights_only=False)

    state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    detected_variant, structure = detect_variant(state)
    merged = vars(cli).copy()
    saved = checkpoint.get("config") if isinstance(checkpoint, dict) else None
    if isinstance(saved, dict):
        runtime_keys = {
            "checkpoint", "eval_annotation", "video_feature_dirs", "text_feature_dir",
            "device", "eval_bsz", "num_workers", "submission_path", "metrics_path",
            "gmiou_threshold", "map_num_workers", "round_to_clip", "allow_partial_load",
            "max_eval_samples", "diagnostic_decoders", "quality_score_alpha",
            "diversity_lambda", "count_exist_threshold",
            "count_confidence_threshold", "window_score_threshold", "save_raw_queries",
        }
        merged = {**merged, **saved, **{key: getattr(cli, key) for key in runtime_keys}}
    configured_variant = merged.get("variant")
    merged["variant"] = detected_variant
    args = finalize_model_arguments(argparse.Namespace(**merged))
    print(json.dumps({
        "checkpoint_structure": structure,
        "detected_variant": detected_variant,
        "configured_variant": configured_variant,
    }, indent=2))

    model, _, device = build_components(args)
    incompatible = model.load_state_dict(state, strict=not args.allow_partial_load)
    if args.allow_partial_load:
        print(json.dumps({
            "missing_keys": list(incompatible.missing_keys),
            "unexpected_keys": list(incompatible.unexpected_keys),
        }, indent=2))

    dataset = make_dataset(
        args, args.eval_annotation, sample_mode="mixed", max_samples=args.max_eval_samples
    )
    loader = make_loader(
        dataset, batch_size=args.eval_bsz, num_workers=args.num_workers, shuffle=False
    )
    decode_modes = ("full",)
    if args.diagnostic_decoders and args.variant in {
        "qd_quality", "qd_dual", "qd_quality_dual", "qd_counter", "qd_hiea2m"
    }:
        decode_modes = ("full", "threshold")
        if args.use_hierarchical_counter:
            decode_modes += ("adaptive",)
    submissions = predict_modes(
        model, loader, device,
        clip_length=args.clip_length, round_to_clip=args.round_to_clip,
        decode_modes=decode_modes,
        quality_score_alpha=args.quality_score_alpha,
        diversity_lambda=args.diversity_lambda,
        count_exist_threshold=args.count_exist_threshold,
        count_confidence_threshold=args.count_confidence_threshold,
        window_score_threshold=args.window_score_threshold,
        save_raw_queries=args.save_raw_queries,
    )
    submission = submissions["full"]

    checkpoint_path = Path(args.checkpoint)
    submission_path = Path(args.submission_path) if args.submission_path else checkpoint_path.with_suffix(".predictions.jsonl")
    metrics_path = Path(args.metrics_path) if args.metrics_path else checkpoint_path.with_suffix(".metrics.json")
    save_jsonl(submission, submission_path)
    metrics = evaluate_submission(
        submission,
        dataset.data,
        gmiou_threshold=args.gmiou_threshold,
        map_num_workers=args.map_num_workers,
    )
    save_json(metrics, metrics_path)
    for mode, diagnostic_submission in submissions.items():
        if mode == "full":
            continue
        diagnostic_submission_path = submission_path.with_name(
            f"{submission_path.stem}.{mode}{submission_path.suffix}"
        )
        diagnostic_metrics_path = metrics_path.with_name(
            f"{metrics_path.stem}.{mode}{metrics_path.suffix}"
        )
        save_jsonl(diagnostic_submission, diagnostic_submission_path)
        diagnostic_metrics = evaluate_submission(
            diagnostic_submission,
            dataset.data,
            gmiou_threshold=args.gmiou_threshold,
            map_num_workers=args.map_num_workers,
        )
        save_json(diagnostic_metrics, diagnostic_metrics_path)
        print(f"{mode}={json.dumps(diagnostic_metrics['brief'], ensure_ascii=False)}")
    print(json.dumps(metrics["brief"], indent=2, ensure_ascii=False))
    print(f"submission={submission_path}")
    print(f"metrics={metrics_path}")


if __name__ == "__main__":
    main()
