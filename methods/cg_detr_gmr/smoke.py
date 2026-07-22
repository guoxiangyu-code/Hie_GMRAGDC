"""Run positive, mixed, and all-null one-step CG-DETR regression smokes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .config import add_data_arguments, add_model_arguments, finalize_model_arguments
from .engine import (
    build_components,
    evaluate_submission,
    make_dataset,
    make_loader,
    predict,
    save_json,
    save_jsonl,
    seed_everything,
    train_one_epoch,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_arguments(parser)
    add_model_arguments(parser)
    parser.set_defaults(
        variant="cg_hiea2m", enc_layers=1, dec_layers=1, t2v_layers=1,
        sent_layers=1, moment_layers=1, dummy_layers=1, num_dummies=3,
        total_prompts=3, num_prompts=1, dim_feedforward=256,
        aux_loss=False, num_workers=0, eval_bsz=4,
    )
    parser.add_argument("--output_dir", default="artifacts/smoke/cg_detr_gmr")
    parser.add_argument("--seed", type=int, default=2018)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = finalize_model_arguments(build_parser().parse_args(argv))
    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model, criterion, device = build_components(args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    summary = {}
    mode_sizes = {"positive": 2, "mixed": 4, "null": 2}
    for mode, size in mode_sizes.items():
        dataset = make_dataset(
            args, args.train_annotation, sample_mode=mode, max_samples=size
        )
        loader = make_loader(dataset, batch_size=size, num_workers=0, shuffle=False)
        losses = train_one_epoch(
            model, criterion, loader, optimizer, device, grad_clip=0.1
        )
        predictions = predict(
            model, loader, device, clip_length=args.clip_length, round_to_clip=True
            , quality_alpha=args.quality_alpha, diversity_lambda=args.diversity_lambda,
            decode_mode="full", existence_threshold=args.existence_threshold,
            count_confidence_threshold=args.count_confidence_threshold,
            window_score_threshold=args.window_score_threshold
        )
        save_jsonl(predictions, output_dir / f"{mode}_predictions.jsonl")
        summary[mode] = {
            "num_examples": len(dataset),
            "num_positive": sum(bool(row.get("relevant_windows", [])) for row in dataset.data),
            "losses": losses,
            "finite": all(torch.isfinite(torch.tensor(value)) for value in losses.values()),
        }
        if mode == "mixed":
            metrics = evaluate_submission(
                predictions, dataset.data, gmiou_threshold=0.4, map_num_workers=1
            )
            summary[mode]["brief"] = metrics["brief"]
            save_json(metrics, output_dir / "mixed_metrics.json")

    # The baseline has no explicit existence adapter; the official evaluator
    # will therefore fall back to its maximum foreground query score.
    baseline_args = argparse.Namespace(**vars(args))
    baseline_args.variant = "cg_detr"
    finalize_model_arguments(baseline_args)
    baseline, baseline_criterion, baseline_device = build_components(baseline_args)
    positive = make_dataset(
        baseline_args, baseline_args.train_annotation, sample_mode="positive", max_samples=2
    )
    positive_loader = make_loader(positive, batch_size=2, num_workers=0, shuffle=False)
    baseline_optimizer = torch.optim.AdamW(baseline.parameters(), lr=1e-4)
    baseline_losses = train_one_epoch(
        baseline, baseline_criterion, positive_loader, baseline_optimizer,
        baseline_device, grad_clip=0.1,
    )
    baseline_predictions = predict(
        baseline, positive_loader, baseline_device,
        clip_length=baseline_args.clip_length, round_to_clip=True,
        quality_alpha=baseline_args.quality_alpha,
        diversity_lambda=baseline_args.diversity_lambda, decode_mode="full",
    )
    summary["baseline_positive"] = {
        "losses": baseline_losses,
        "has_explicit_existence": any("pred_exist_score" in row for row in baseline_predictions),
    }

    save_json(summary, output_dir / "summary.json")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
