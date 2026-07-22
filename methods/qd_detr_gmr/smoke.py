"""Run positive, mixed, and all-null one-step QD-DETR regression smokes."""

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
    weighted_loss,
)
from .dataset import prepare_batch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_arguments(parser)
    add_model_arguments(parser)
    parser.set_defaults(
        variant="qd_hiea2m", enc_layers=1, dec_layers=1,
        dim_feedforward=256, aux_loss=False, num_workers=0, eval_bsz=4,
    )
    parser.add_argument("--output_dir", default="artifacts/smoke/qd_detr_gmr")
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
    baseline_args.variant = "qd_detr"
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
    )
    summary["baseline_positive"] = {
        "losses": baseline_losses,
        "has_explicit_existence": any("pred_exist_score" in row for row in baseline_predictions),
    }

    # Two-stage warm-start audit: train the parent existence adapter on a
    # positive batch, migrate its shared weights, then verify the HieA2M heads
    # neither disturb localization nor collapse conditional counting on nulls.
    parent_args = argparse.Namespace(**vars(args))
    parent_args.variant = "qd_detr_gmr"
    finalize_model_arguments(parent_args)
    parent, parent_criterion, parent_device = build_components(parent_args)
    parent_optimizer = torch.optim.AdamW(parent.parameters(), lr=1e-4)
    stage1_positive = make_dataset(
        parent_args, parent_args.train_annotation, sample_mode="positive", max_samples=2
    )
    stage1_loader = make_loader(stage1_positive, batch_size=2, num_workers=0, shuffle=False)
    stage1_losses = train_one_epoch(
        parent, parent_criterion, stage1_loader, parent_optimizer,
        parent_device, grad_clip=0.1,
    )

    child_args = argparse.Namespace(**vars(args))
    child_args.variant = "qd_hiea2m"
    finalize_model_arguments(child_args)
    child, child_criterion, child_device = build_components(child_args)
    incompatible = child.load_state_dict(parent.state_dict(), strict=False)
    mixed_dataset = make_dataset(
        child_args, child_args.train_annotation, sample_mode="mixed", max_samples=4
    )
    mixed_loader = make_loader(mixed_dataset, batch_size=4, num_workers=0, shuffle=False)
    mixed_batch = next(iter(mixed_loader))
    _, mixed_inputs, mixed_targets = prepare_batch(mixed_batch, child_device)
    parent.eval()
    child.eval()
    with torch.no_grad():
        parent_outputs = parent(**mixed_inputs)
        warm_outputs = child(**mixed_inputs)
    localization_delta = max(
        float((parent_outputs["pred_logits"] - warm_outputs["pred_logits"]).abs().max()),
        float((parent_outputs["pred_spans"] - warm_outputs["pred_spans"]).abs().max()),
    )
    existence_delta = float(
        (parent_outputs["pred_exist_logits"] - warm_outputs["pred_exist_logits"]).abs().max()
    )

    child.train()
    child.zero_grad(set_to_none=True)
    mixed_outputs = child(**mixed_inputs)
    mixed_loss_dict = child_criterion(mixed_outputs, mixed_targets)
    mixed_total = weighted_loss(mixed_loss_dict, child_criterion.weight_dict)
    mixed_total.backward()
    count_gradient = child.hierarchical_counter.count_head.weight.grad
    mixed_count_gradient_norm = 0.0 if count_gradient is None else float(count_gradient.norm())
    quality_gradient = child.quality_embed.layers[-1].weight.grad
    sentence_gate_gradient = child.dual_grounding.sentence_gate_logit.grad
    phrase_gate_gradient = child.dual_grounding.phrase_gate_logit.grad
    mixed_new_module_gradients = {
        "quality": 0.0 if quality_gradient is None else float(quality_gradient.norm()),
        "dual_sentence_gate": (
            0.0 if sentence_gate_gradient is None else float(sentence_gate_gradient.abs())
        ),
        "dual_phrase_gate": (
            0.0 if phrase_gate_gradient is None else float(phrase_gate_gradient.abs())
        ),
        "positive_count": mixed_count_gradient_norm,
    }

    null_dataset = make_dataset(
        child_args, child_args.train_annotation, sample_mode="null", max_samples=2
    )
    null_loader = make_loader(null_dataset, batch_size=2, num_workers=0, shuffle=False)
    _, null_inputs, null_targets = prepare_batch(next(iter(null_loader)), child_device)
    child.zero_grad(set_to_none=True)
    null_outputs = child(**null_inputs)
    null_loss_dict = child_criterion(null_outputs, null_targets)
    null_total = weighted_loss(null_loss_dict, child_criterion.weight_dict)
    null_total.backward()
    null_count_gradient = child.hierarchical_counter.count_head.weight.grad
    null_count_gradient_norm = (
        0.0 if null_count_gradient is None else float(null_count_gradient.norm())
    )
    counter_exist_gradient = child.hierarchical_counter.exist_head.weight.grad
    summary["two_stage_noncollapse"] = {
        "stage1_losses": stage1_losses,
        "missing_new_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
        "warmstart_localization_max_delta": localization_delta,
        "warmstart_existence_max_delta": existence_delta,
        "warmstart_residuals": {
            "dual_sentence_gate": float(warm_outputs["dual_sentence_gate"]),
            "dual_phrase_gate": float(warm_outputs["dual_phrase_gate"]),
            "quality_logit_max_abs": float(warm_outputs["pred_quality_logits"].abs().max()),
            "counter_exist_delta_max_abs": float(
                warm_outputs["pred_counter_exist_delta"].abs().max()
            ),
        },
        "mixed_positive_count_gradient_norm": mixed_count_gradient_norm,
        "mixed_new_module_gradient_norms": mixed_new_module_gradients,
        "null_conditional_losses": {
            name: float(null_loss_dict[name].detach())
            for name in (
                "loss_count", "loss_count_ordinal", "loss_count_contrastive",
                "loss_count_consistency",
            )
        },
        "null_count_head_gradient_norm": null_count_gradient_norm,
        "null_existence_gradient_norm": (
            0.0 if counter_exist_gradient is None else float(counter_exist_gradient.norm())
        ),
        "noncollapse_pass": (
            localization_delta < 1e-6
            and existence_delta < 1e-6
            and mixed_count_gradient_norm > 0
            and null_count_gradient_norm == 0
            and counter_exist_gradient is not None
        ),
    }

    save_json(summary, output_dir / "summary.json")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
