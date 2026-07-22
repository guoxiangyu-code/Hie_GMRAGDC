"""Train QD-DETR or QD-DETR-GMR on Soccer-GMR precomputed features."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from models.moment_detr_gmr.hierarchical_counter import inverse_sqrt_positive_count_weights
from methods.resume_state import make_training_state, restore_training_state

from .config import add_data_arguments, add_model_arguments, finalize_model_arguments
from .engine import (
    build_components,
    evaluate_submission,
    harmonic_joint,
    load_jsonl,
    make_dataset,
    make_loader,
    predict_modes,
    save_json,
    save_jsonl,
    seed_everything,
    train_one_epoch,
)


STAGE_NEW_PREFIXES = {
    "qd_detr": (),
    "qd_detr_gmr": ("exist_head.",),
    "qd_quality": ("quality_embed.",),
    "qd_dual": ("dual_grounding.",),
    "qd_counter": ("hierarchical_counter.",),
    "qd_hiea2m": ("quality_embed.", "dual_grounding.", "hierarchical_counter."),
}

RESUME_MUTABLE_KEYS = {
    "resume", "init_checkpoint", "epochs", "num_workers", "eval_bsz",
    "map_num_workers", "device",
}


def validate_resume_config(checkpoint: dict, args: argparse.Namespace) -> None:
    """Reject a nominal full resume whose training semantics have changed."""
    saved = checkpoint.get("config")
    if not isinstance(saved, dict):
        raise ValueError("resume checkpoint is missing its saved config")
    current = vars(args)
    keys = (set(saved) & set(current)) - RESUME_MUTABLE_KEYS
    # Old checkpoints predate this switch and therefore mean legacy ``False``.
    keys.add("mask_null_vmr_loss")
    mismatches = {}
    for key in sorted(keys):
        saved_value = saved.get(key, False if key == "mask_null_vmr_loss" else None)
        current_value = current.get(key, False if key == "mask_null_vmr_loss" else None)
        if saved_value != current_value:
            mismatches[key] = {"checkpoint": saved_value, "cli": current_value}
    if mismatches:
        raise ValueError(f"resume config mismatch: {mismatches}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_arguments(parser)
    add_model_arguments(parser)
    parser.add_argument("--output_dir", default="artifacts/qd_detr_gmr/run")
    parser.add_argument("--seed", type=int, default=2018)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--backbone_lr_scale", type=float, default=0.1)
    parser.add_argument("--lr_drop", type=int, default=400)
    parser.add_argument("--grad_clip", type=float, default=0.1)
    parser.add_argument("--eval_interval", type=int, default=1)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument(
        "--train_sample_mode", choices=("auto", "positive", "mixed", "null"), default="auto",
        help="auto: positive for qd_detr; mixed for qd_detr_gmr",
    )
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--init_checkpoint", default=None, help="weights-only initialization")
    parser.add_argument("--resume", default=None, help="resume this runner's full checkpoint")
    parser.add_argument("--gmiou_threshold", type=float, default=0.4)
    parser.add_argument("--reference_map", type=float, default=0.0)
    parser.add_argument("--reference_gmiou3", type=float, default=0.0)
    parser.add_argument("--map_num_workers", type=int, default=1)
    parser.add_argument("--round_to_clip", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--diagnostic_decoders", action=argparse.BooleanOptionalAction, default=True)
    return parser


def _load_weights(model, checkpoint_path: str, allowed_new_prefixes: tuple[str, ...]) \
        -> tuple[list[str], list[str]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    incompatible = model.load_state_dict(state, strict=False)
    whole_new_prefixes = tuple(
        prefix for prefix in allowed_new_prefixes
        if not any(name.startswith(prefix) for name in state)
    )
    invalid_missing = [
        name for name in incompatible.missing_keys
        if not name.startswith(whole_new_prefixes)
    ]
    unexpected = list(incompatible.unexpected_keys)
    if invalid_missing or unexpected:
        raise RuntimeError(
            "unsafe parent checkpoint migration: "
            f"missing={invalid_missing}, unexpected={unexpected}"
        )
    return list(incompatible.missing_keys), unexpected


def _save_checkpoint(
    path: Path, model, optimizer, scheduler, epoch: int, args, metrics: dict,
    *, training_state: dict | None = None,
) -> None:
    payload = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "config": vars(args),
            "metrics": metrics,
            "source_revision": "f8628f79f7c651b586300b142dbe9b85e43857cc",
        }
    if training_state is not None:
        payload["training_state"] = training_state
    torch.save(payload, path)


def main(argv: list[str] | None = None) -> None:
    args = finalize_model_arguments(build_parser().parse_args(argv))
    if args.resume and args.init_checkpoint:
        raise ValueError("--resume and --init_checkpoint are mutually exclusive")
    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_mode = args.train_sample_mode
    if sample_mode == "auto":
        sample_mode = "positive" if args.variant == "qd_detr" else "mixed"
    train_dataset = make_dataset(
        args, args.train_annotation, sample_mode=sample_mode, max_samples=args.max_train_samples
    )
    eval_dataset = make_dataset(
        args, args.eval_annotation, sample_mode="mixed", max_samples=args.max_eval_samples
    )
    positive_count_histogram = [0, 0, 0, 0]
    for row in train_dataset.data:
        count = len(row.get("relevant_windows", []) or [])
        if count > 0:
            positive_count_histogram[min(count, 4) - 1] += 1
    args.positive_count_weights = inverse_sqrt_positive_count_weights(
        positive_count_histogram
    ).tolist()
    args.positive_count_histogram = positive_count_histogram
    resume_checkpoint = None
    if args.resume:
        resume_checkpoint = torch.load(
            args.resume, map_location="cpu", weights_only=False
        )
        validate_resume_config(resume_checkpoint, args)
    save_json(vars(args), output_dir / "config.json")
    train_loader = make_loader(
        train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True
    )
    eval_loader = make_loader(
        eval_dataset, batch_size=args.eval_bsz, num_workers=args.num_workers, shuffle=False
    )

    model, criterion, device = build_components(args)
    stage_new_prefixes = STAGE_NEW_PREFIXES[args.variant]
    missing_stage_prefixes = [
        prefix for prefix in stage_new_prefixes
        if not any(
            name.startswith(prefix) and parameter.requires_grad
            for name, parameter in model.named_parameters()
        )
    ]
    if missing_stage_prefixes:
        raise RuntimeError(
            f"optimizer stage prefixes do not match trainable parameters: {missing_stage_prefixes}"
        )
    new_parameters = [
        parameter for name, parameter in model.named_parameters()
        if parameter.requires_grad and name.startswith(stage_new_prefixes)
    ]
    shared_parameters = [
        parameter for name, parameter in model.named_parameters()
        if parameter.requires_grad and not name.startswith(stage_new_prefixes)
    ]
    if new_parameters:
        parameter_groups = [
            {"params": shared_parameters, "lr": args.lr * args.backbone_lr_scale},
            {"params": new_parameters, "lr": args.lr},
        ]
    else:
        parameter_groups = [{"params": model.parameters(), "lr": args.lr}]
    save_json(
        {
            "variant": args.variant,
            "stage_new_prefixes": list(stage_new_prefixes),
            "groups": [
                {
                    "role": "shared" if index == 0 else "new",
                    "lr": float(group["lr"]),
                    "num_parameters": int(sum(parameter.numel() for parameter in group["params"])),
                }
                for index, group in enumerate(parameter_groups)
            ],
        },
        output_dir / "optimizer_groups.json",
    )
    optimizer = torch.optim.AdamW(parameter_groups, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_drop, gamma=0.1)
    start_epoch = 0

    if args.init_checkpoint:
        allowed_new_prefixes = []
        if args.use_exist_head:
            allowed_new_prefixes.append("exist_head.")
        if args.use_quality_head:
            allowed_new_prefixes.append("quality_embed.")
        if args.use_dual_grounding:
            allowed_new_prefixes.append("dual_grounding.")
        if args.use_hierarchical_counter:
            allowed_new_prefixes.append("hierarchical_counter.")
        missing, unexpected = _load_weights(
            model, args.init_checkpoint, tuple(allowed_new_prefixes)
        )
        audit = {"missing_keys": missing, "unexpected_keys": unexpected}
        save_json(audit, output_dir / "initialization_audit.json")
        print(json.dumps({"initialization_audit": audit}, indent=2))
    if args.resume:
        checkpoint = resume_checkpoint
        assert checkpoint is not None
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1

    log_path = output_dir / "train_log.jsonl"
    ground_truth = eval_dataset.data
    best = {"mAP": float("-inf"), "G-mIoU@3": float("-inf"), "joint": float("-inf")}
    epochs_without_improvement = 0
    if args.resume:
        assert resume_checkpoint is not None
        primary_name = "mAP" if args.variant == "qd_detr" else "joint"
        best, epochs_without_improvement = restore_training_state(
            resume_checkpoint, log_path, primary_metric=primary_name
        )
    for epoch in range(start_epoch, args.epochs):
        losses = train_one_epoch(
            model, criterion, train_loader, optimizer, device, grad_clip=args.grad_clip
        )
        scheduler.step()
        record: dict = {"epoch": epoch + 1, "train": losses}

        if (epoch + 1) % args.eval_interval == 0:
            decode_modes = ("full",)
            if args.diagnostic_decoders and args.variant in {
                "qd_quality", "qd_dual", "qd_counter", "qd_hiea2m"
            }:
                decode_modes = ("full", "threshold")
                if args.use_hierarchical_counter:
                    decode_modes += ("adaptive",)
            submissions = predict_modes(
                model, eval_loader, device,
                clip_length=args.clip_length, round_to_clip=args.round_to_clip,
                decode_modes=decode_modes,
                quality_score_alpha=args.quality_score_alpha,
                diversity_lambda=args.diversity_lambda,
                count_exist_threshold=args.count_exist_threshold,
                count_confidence_threshold=args.count_confidence_threshold,
                window_score_threshold=args.window_score_threshold,
            )
            submission = submissions["full"]
            prediction_path = output_dir / "latest_val_predictions.jsonl"
            save_jsonl(submission, prediction_path)
            metrics = evaluate_submission(
                submission,
                ground_truth,
                gmiou_threshold=args.gmiou_threshold,
                map_num_workers=args.map_num_workers,
            )
            save_json(metrics, output_dir / "latest_val_metrics.json")
            brief = metrics["brief"]
            scores = {
                "mAP": float(brief["mAP"]),
                "G-mIoU@3": float(brief["G-mIoU@3"]),
            }
            if args.reference_map > 0 and args.reference_gmiou3 > 0:
                scores["joint"] = min(
                    scores["mAP"] / args.reference_map,
                    scores["G-mIoU@3"] / args.reference_gmiou3,
                )
            else:
                scores["joint"] = harmonic_joint(scores["mAP"], scores["G-mIoU@3"])
            record["val"] = brief
            record["selection"] = scores
            if len(submissions) > 1:
                record["diagnostic_decoders"] = {}
                for mode, diagnostic_submission in submissions.items():
                    if mode == "full":
                        continue
                    diagnostic_prediction_path = output_dir / f"latest_val_{mode}_predictions.jsonl"
                    save_jsonl(diagnostic_submission, diagnostic_prediction_path)
                    diagnostic_metrics = evaluate_submission(
                        diagnostic_submission,
                        ground_truth,
                        gmiou_threshold=args.gmiou_threshold,
                        map_num_workers=args.map_num_workers,
                    )
                    save_json(
                        diagnostic_metrics,
                        output_dir / f"latest_val_{mode}_metrics.json",
                    )
                    record["diagnostic_decoders"][mode] = diagnostic_metrics["brief"]

            improved_primary = False
            for metric_name, score in scores.items():
                if score <= best[metric_name]:
                    continue
                best[metric_name] = score
                suffix = metric_name.lower().replace("-", "_").replace("@", "")
                checkpoint_path = output_dir / f"best_{suffix}.ckpt"
                _save_checkpoint(
                    checkpoint_path, model, optimizer, scheduler, epoch, args, metrics
                )
                shutil.copy2(prediction_path, output_dir / f"best_{suffix}_val_predictions.jsonl")
                save_json(metrics, output_dir / f"best_{suffix}_val_metrics.json")
                for mode in submissions:
                    if mode == "full":
                        continue
                    shutil.copy2(
                        output_dir / f"latest_val_{mode}_predictions.jsonl",
                        output_dir / f"best_{suffix}_val_{mode}_predictions.jsonl",
                    )
                    shutil.copy2(
                        output_dir / f"latest_val_{mode}_metrics.json",
                        output_dir / f"best_{suffix}_val_{mode}_metrics.json",
                    )
                primary_name = "mAP" if args.variant == "qd_detr" else "joint"
                if metric_name == primary_name:
                    shutil.copy2(checkpoint_path, output_dir / "best.ckpt")
                    improved_primary = True
            epochs_without_improvement = 0 if improved_primary else epochs_without_improvement + 1
            _save_checkpoint(
                output_dir / "latest.ckpt", model, optimizer, scheduler, epoch, args,
                metrics,
                training_state=make_training_state(
                    best, epochs_without_improvement
                ),
            )

        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps(record, ensure_ascii=False))
        if args.patience >= 0 and epochs_without_improvement > args.patience:
            print(f"early stop after epoch {epoch + 1}")
            break


if __name__ == "__main__":
    main()
