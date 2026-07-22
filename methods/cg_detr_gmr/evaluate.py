"""Evaluate a CG-DETR(-GMR) checkpoint with the official Soccer-GMR metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .config import (
    REPO_ROOT,
    add_data_arguments,
    add_model_arguments,
    detect_variant,
    finalize_model_arguments,
)
from .engine import (
    build_components,
    evaluate_submission,
    make_dataset,
    make_loader,
    predict,
    save_json,
    save_jsonl,
)


RUNTIME_KEYS = {
    "checkpoint", "eval_annotation", "video_feature_dirs", "text_feature_dir",
    "device", "eval_bsz", "num_workers", "submission_path", "metrics_path",
    "gmiou_threshold", "map_num_workers", "round_to_clip", "allow_partial_load",
    "max_eval_samples", "save_raw_queries",
}
DECODER_OVERRIDE_DEFAULTS = {
    "decode_mode": "full",
    "existence_threshold": 0.4,
    "count_confidence_threshold": 0.55,
    "window_score_threshold": 0.1,
    "quality_alpha": 0.5,
    "diversity_lambda": 0.0,
}


def merge_checkpoint_args(
    cli: argparse.Namespace, saved: dict | None,
) -> dict:
    """Let explicit evaluation controls override a checkpoint configuration.

    Decoder controls use ``None`` as the CLI sentinel so an omitted option
    preserves a tuned value stored in the checkpoint.  Older/raw checkpoints
    without a saved configuration receive the canonical defaults.
    """
    merged = vars(cli).copy()
    if isinstance(saved, dict):
        runtime = {key: getattr(cli, key) for key in RUNTIME_KEYS}
        for key, default in DECODER_OVERRIDE_DEFAULTS.items():
            value = getattr(cli, key)
            if value is not None:
                runtime[key] = value
            elif key not in saved:
                runtime[key] = default
        return {**merged, **saved, **runtime}
    for key, default in DECODER_OVERRIDE_DEFAULTS.items():
        if merged.get(key) is None:
            merged[key] = default
    return merged


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
    parser.add_argument("--save_raw_queries", action="store_true")
    parser.set_defaults(**{key: None for key in DECODER_OVERRIDE_DEFAULTS})
    return parser


def main(argv: list[str] | None = None) -> None:
    cli = build_parser().parse_args(argv)
    checkpoint = torch.load(cli.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    detected_variant, checkpoint_structure = detect_variant(state)

    saved = checkpoint.get("config") if isinstance(checkpoint, dict) else None
    merged = merge_checkpoint_args(cli, saved)
    saved_variant = merged.get("variant")
    merged["variant"] = detected_variant
    args = finalize_model_arguments(argparse.Namespace(**merged))
    if saved_variant != detected_variant:
        print(json.dumps({
            "checkpoint_structure_override": {
                "saved_variant": saved_variant,
                "detected_variant": detected_variant,
                "structure": checkpoint_structure,
            }
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
    submission = predict(
        model, loader, device,
        clip_length=args.clip_length, round_to_clip=args.round_to_clip,
        quality_alpha=args.quality_alpha,
        diversity_lambda=args.diversity_lambda,
        decode_mode=args.decode_mode,
        existence_threshold=args.existence_threshold,
        count_confidence_threshold=args.count_confidence_threshold,
        window_score_threshold=args.window_score_threshold,
        save_raw_queries=args.save_raw_queries,
    )

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
    print(json.dumps(metrics["brief"], indent=2, ensure_ascii=False))
    print(f"submission={submission_path}")
    print(f"metrics={metrics_path}")


if __name__ == "__main__":
    main()
