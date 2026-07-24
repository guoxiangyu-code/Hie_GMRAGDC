"""Train the isolated EaTR baseline or EaTR-GMR existence adapter.

Run as ``python -m methods.eatr_gmr.train --help`` from the repository root.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_
from methods.resume_state import make_training_state, restore_training_state
from torch.utils.data import DataLoader

from .build import build_model
from .checkpoint import config_from_checkpoint, load_parent_state
from .cli import add_data_arguments, add_model_arguments, add_runtime_arguments
from .config import EaTRConfig
from .dataset import SoccerGMRDataset, collate_fn, move_batch
from .runtime import (
    official_metrics,
    predict_views,
    resolve_device,
    seed_everything,
    write_json,
    write_jsonl,
)
from .variants import VARIANT_FLAGS, apply_variant


STAGE_NEW_PREFIXES = {
    "eatr": (),
    "eatr_gmr": ("exist_head.",),
    "eatr_quality": ("quality_embed.",),
    "eatr_dual": ("dual_grounding.",),
    "eatr_quality_dual": ("quality_embed.", "dual_grounding."),
    "eatr_counter": ("hierarchical_counter.",),
    "eatr_hiea2m": ("quality_embed.", "dual_grounding.", "hierarchical_counter."),
}


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train EaTR/EaTR-GMR on Soccer-GMR")
    add_data_arguments(parser, require_train=True)
    add_model_arguments(parser)
    add_runtime_arguments(parser)
    parser.add_argument("--variant", choices=tuple(VARIANT_FLAGS), default="eatr_gmr")
    parser.add_argument("--val-annotations")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--eval-interval", type=int, default=1)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lr-drop", type=int, default=150)
    parser.add_argument("--grad-clip", type=float, default=0.1)
    parser.add_argument("--exist-loss-coef", type=float, default=1.0)
    parser.add_argument("--quality-loss-coef", type=float, default=1.0)
    parser.add_argument("--quality-score-alpha", type=float, default=0.5)
    parser.add_argument(
        "--mask-null-vmr-loss",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "apply the GMR indicator I(y=1) to localization, event, and "
            "query-quality losses for null queries"
        ),
    )
    parser.add_argument("--diversity-lambda", type=float, default=0.0)
    parser.add_argument("--dual-num-phrases", type=int, default=3)
    parser.add_argument("--dual-num-dummies", type=int, default=3)
    parser.add_argument("--dual-slot-iterations", type=int, default=1)
    parser.add_argument("--dual-gate-init", type=float, default=-4.0)
    parser.add_argument("--dual-dqa-loss-coef", type=float, default=0.05)
    parser.add_argument("--dual-eos-loss-coef", type=float, default=0.1)
    parser.add_argument("--counter-dropout", type=float, default=0.1)
    parser.add_argument("--count-loss-coef", type=float, default=1.0)
    parser.add_argument("--count-ordinal-loss-coef", type=float, default=0.25)
    parser.add_argument("--count-contrastive-loss-coef", type=float, default=0.05)
    parser.add_argument("--count-consistency-loss-coef", type=float, default=0.05)
    parser.add_argument(
        "--positive-count-class-counts", type=int, nargs=4,
        default=[1423, 565, 117, 31],
        metavar=("N1", "N2", "N3", "N4PLUS"),
    )
    parser.add_argument("--count-exist-threshold", type=float, default=0.4)
    parser.add_argument("--count-confidence-threshold", type=float, default=0.55)
    parser.add_argument("--window-score-threshold", type=float, default=0.1)
    parser.add_argument("--diagnostic-mode", choices=("adaptive", "hard"), default="adaptive")
    parser.add_argument("--backbone-lr-scale", type=float, default=0.1)
    parser.add_argument("--reference-map", type=float, default=0.0)
    parser.add_argument("--reference-gmiou3", type=float, default=0.0)
    parser.add_argument(
        "--train-sample-mode",
        choices=("auto", "positive", "mixed", "null"),
        default="auto",
        help="auto uses positive-only for eatr and mixed queries for GMR variants",
    )
    parser.add_argument("--resume")
    parser.add_argument(
        "--init-checkpoint",
        help="Warm-start a larger variant from an EaTR/EaTR-GMR parent checkpoint",
    )
    parser.add_argument("--max-train-steps", type=int, default=None)
    return parser


def config_from_args(args: argparse.Namespace) -> EaTRConfig:
    config = EaTRConfig(
        video_dim=args.video_feature_dim + 2,
        text_dim=args.text_feature_dim,
        hidden_dim=args.hidden_dim,
        nheads=args.nheads,
        enc_layers=args.enc_layers,
        dec_layers=args.dec_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        input_dropout=args.input_dropout,
        num_queries=args.num_queries,
        num_slot_iter=args.num_slot_iter,
        n_input_proj=args.n_input_proj,
        max_q_l=args.max_text_len,
        max_v_l=args.max_video_len,
        use_txt_pos=args.use_text_position,
        aux_loss=not args.no_aux_loss,
        exist_loss_coef=args.exist_loss_coef,
        quality_loss_coef=args.quality_loss_coef,
        quality_score_alpha=args.quality_score_alpha,
        mask_null_vmr_loss=args.mask_null_vmr_loss,
        diversity_lambda=args.diversity_lambda,
        dual_num_phrases=args.dual_num_phrases,
        dual_num_dummies=args.dual_num_dummies,
        dual_slot_iterations=args.dual_slot_iterations,
        dual_gate_init=args.dual_gate_init,
        dual_dqa_loss_coef=args.dual_dqa_loss_coef,
        dual_eos_loss_coef=args.dual_eos_loss_coef,
        counter_dropout=args.counter_dropout,
        count_loss_coef=args.count_loss_coef,
        count_ordinal_loss_coef=args.count_ordinal_loss_coef,
        count_contrastive_loss_coef=args.count_contrastive_loss_coef,
        count_consistency_loss_coef=args.count_consistency_loss_coef,
        positive_count_class_counts=tuple(args.positive_count_class_counts),
        count_exist_threshold=args.count_exist_threshold,
        count_confidence_threshold=args.count_confidence_threshold,
        window_score_threshold=args.window_score_threshold,
    )
    return apply_variant(config, args.variant)


def make_dataset(args: argparse.Namespace, annotation_path: str, *, load_labels: bool,
                 sample_mode: str = "mixed"):
    dataset = SoccerGMRDataset(
        annotation_path=annotation_path,
        slowfast_dir=args.slowfast_dir,
        clip_dir=args.clip_dir,
        text_dir=args.text_dir,
        max_video_len=args.max_video_len,
        max_text_len=args.max_text_len,
        clip_length=args.clip_length,
        max_windows=args.max_windows,
        load_labels=load_labels,
        trim_text_by_attention_mask=args.trim_text_by_attention_mask,
        expected_video_feature_dim=args.video_feature_dim,
        expected_text_feature_dim=args.text_feature_dim,
    )
    if sample_mode == "positive":
        dataset.data = [row for row in dataset.data if row.get("relevant_windows")]
    elif sample_mode == "null":
        dataset.data = [row for row in dataset.data if not row.get("relevant_windows")]
    elif sample_mode != "mixed":
        raise ValueError(f"unknown sample_mode={sample_mode!r}")
    if not dataset.data:
        raise ValueError(f"sample_mode={sample_mode!r} produced an empty dataset")
    return dataset


def train_one_epoch(model, criterion, loader, optimizer, device: torch.device,
                    grad_clip: float, max_steps: int | None = None) -> dict[str, float]:
    model.train()
    criterion.train()
    sums: dict[str, float] = {}
    steps = 0
    for _, model_inputs, targets in loader:
        model_inputs, targets = move_batch(model_inputs, targets, device)
        outputs = model(**model_inputs)
        loss_dict = criterion(outputs, targets)
        loss = criterion.weighted_loss(loss_dict)
        if not torch.isfinite(loss):
            values = {name: float(value.detach().cpu()) for name, value in loss_dict.items()}
            raise FloatingPointError(f"non-finite loss: {values}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0:
            clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        steps += 1
        sums["loss_total"] = sums.get("loss_total", 0.0) + float(loss.detach().cpu())
        for name, value in loss_dict.items():
            sums[name] = sums.get(name, 0.0) + float(value.detach().cpu())
        if max_steps is not None and steps >= max_steps:
            break
    if steps == 0:
        raise ValueError("training loader produced no batches")
    return {name: value / steps for name, value in sums.items()}


def save_checkpoint(path: Path, model, optimizer, scheduler, config: EaTRConfig,
                    args: argparse.Namespace, epoch: int, metrics=None,
                    *, training_state: dict | None = None) -> None:
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "config": config.to_dict(),
        "args": vars(args),
        "epoch": epoch,
        "metrics": metrics,
        "upstream_commit": "384f09396a32741a73106d2e147bde54cbcce48f",
    }
    if training_state is not None:
        payload["training_state"] = training_state
    torch.save(payload, path)


def joint_score(map_score: float, gmiou3_score: float,
                reference_map: float = 0.0,
                reference_gmiou3: float = 0.0) -> float:
    """Joint selection that cannot hide a regression behind one large gain."""
    if reference_map > 0 and reference_gmiou3 > 0:
        return min(map_score / reference_map, gmiou3_score / reference_gmiou3)
    return 2.0 * map_score * gmiou3_score / max(map_score + gmiou3_score, 1e-8)


def main(argv=None) -> None:
    args = make_parser().parse_args(argv)
    if args.resume and args.init_checkpoint:
        raise ValueError("--resume and --init-checkpoint are mutually exclusive")
    seed_everything(args.seed)
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = config_from_args(args)
    model, criterion = build_model(config)
    initialization = None
    if args.init_checkpoint:
        parent = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        parent_state = parent.get("model", parent)
        missing = load_parent_state(model, parent_state)
        _, parent_structure = config_from_checkpoint(parent)
        initialization = {
            "checkpoint": str(Path(args.init_checkpoint).resolve()),
            "parent_structure": parent_structure,
            "initialized_new_keys": missing,
        }
    model.to(device)
    criterion.to(device)
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
    new_parameters = []
    shared_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        target = new_parameters if name.startswith(stage_new_prefixes) else shared_parameters
        target.append(parameter)
    if new_parameters:
        optimizer_groups = [
            {"params": shared_parameters, "lr": args.lr * args.backbone_lr_scale},
            {"params": new_parameters, "lr": args.lr},
        ]
    else:
        optimizer_groups = [{"params": shared_parameters, "lr": args.lr}]
    optimizer = torch.optim.AdamW(
        optimizer_groups, lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_drop, gamma=0.1)
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        resumed_config, resumed_structure = config_from_checkpoint(checkpoint)
        if resumed_config != config:
            raise ValueError(
                "resume checkpoint model config differs from CLI config: "
                f"detected {resumed_structure}"
            )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1

    train_sample_mode = args.train_sample_mode
    if train_sample_mode == "auto":
        train_sample_mode = "positive" if args.variant == "eatr" else "mixed"
    train_dataset = make_dataset(
        args,
        args.train_annotations,
        load_labels=True,
        sample_mode=train_sample_mode,
    )
    observed_positive_count_histogram = [0, 0, 0, 0]
    for row in train_dataset.data:
        count = len(row.get("relevant_windows", []) or [])
        if count > 0:
            observed_positive_count_histogram[min(count, 4) - 1] += 1
    args.observed_positive_count_histogram = observed_positive_count_histogram
    if (
        config.use_hierarchical_counter
        and train_sample_mode != "null"
        and tuple(observed_positive_count_histogram)
        != tuple(config.positive_count_class_counts)
    ):
        raise ValueError(
            "configured positive count class counts do not match the training split: "
            f"configured={config.positive_count_class_counts}, "
            f"observed={observed_positive_count_histogram}"
        )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )
    val_dataset = None
    val_loader = None
    if args.val_annotations:
        val_dataset = make_dataset(
            args, args.val_annotations, load_labels=True, sample_mode="mixed"
        )
        val_loader = DataLoader(
            val_dataset, batch_size=args.eval_batch_size, shuffle=False,
            num_workers=args.num_workers, collate_fn=collate_fn,
            pin_memory=device.type == "cuda",
        )

    optimizer_audit = {
        "variant": args.variant,
        "stage_new_prefixes": list(stage_new_prefixes),
        "groups": [
            {
                "role": "shared" if index == 0 else "new",
                "lr": float(group["lr"]),
                "num_parameters": int(sum(parameter.numel() for parameter in group["params"])),
            }
            for index, group in enumerate(optimizer_groups)
        ],
    }
    write_json(
        {
            "config": config.to_dict(),
            "args": vars(args),
            "initialization": initialization,
            "optimizer_groups": optimizer_audit,
        },
        output_dir / "run.json",
    )
    log_path = output_dir / "train_log.jsonl"
    best = {"mAP": -math.inf, "G-mIoU@3": -math.inf, "joint": -math.inf}
    epochs_without_improvement = 0
    if args.resume:
        primary_name = "mAP" if args.variant == "eatr" else "joint"
        best, epochs_without_improvement = restore_training_state(
            checkpoint, log_path, primary_metric=primary_name
        )
    for epoch in range(start_epoch, args.epochs):
        losses = train_one_epoch(
            model, criterion, train_loader, optimizer, device,
            grad_clip=args.grad_clip, max_steps=args.max_train_steps,
        )
        scheduler.step()
        metrics = None
        primary_metrics = None
        record = {"epoch": epoch + 1, "train": losses}
        evaluated = val_loader is not None and (epoch + 1) % args.eval_interval == 0
        if evaluated:
            modes = (
                ("full", args.diagnostic_mode)
                if config.use_hierarchical_counter else ("full",)
            )
            views = predict_views(
                model, val_loader, device, modes=modes, max_predictions=10,
                round_to_clip=True, clip_length=args.clip_length,
                quality_alpha=args.quality_score_alpha,
                diversity_lambda=args.diversity_lambda,
                existence_threshold=args.count_exist_threshold,
                count_confidence_threshold=args.count_confidence_threshold,
                window_score_threshold=args.window_score_threshold,
            )
            submission = views["full"]
            full_submission_path = output_dir / "latest_val_predictions.jsonl"
            full_metrics_path = output_dir / "latest_val_metrics.json"
            write_jsonl(submission, full_submission_path)
            metrics = official_metrics(submission, val_dataset.data, map_num_workers=1)
            write_json(metrics, full_metrics_path)
            if config.use_hierarchical_counter:
                diagnostic_submission = views[args.diagnostic_mode]
                diagnostic_metrics = official_metrics(
                    diagnostic_submission, val_dataset.data, map_num_workers=1
                )
                write_jsonl(
                    diagnostic_submission,
                    output_dir / f"latest_val_{args.diagnostic_mode}_predictions.jsonl",
                )
                write_json(
                    diagnostic_metrics,
                    output_dir / f"latest_val_{args.diagnostic_mode}_metrics.json",
                )
                metrics = {"full": metrics, args.diagnostic_mode: diagnostic_metrics}
                primary_metrics = metrics["full"]
            else:
                primary_metrics = metrics
            map_score = float(primary_metrics["brief"]["mAP"])
            gmiou3_score = float(primary_metrics["brief"]["G-mIoU@3"])
            scores = {
                "mAP": map_score,
                "G-mIoU@3": gmiou3_score,
                "joint": joint_score(
                    map_score,
                    gmiou3_score,
                    args.reference_map,
                    args.reference_gmiou3,
                ),
            }
            record["val"] = primary_metrics["brief"]
            record["selection"] = scores
            primary_name = "mAP" if args.variant == "eatr" else "joint"
            improved_primary = False
            for metric_name, score in scores.items():
                if score <= best[metric_name]:
                    continue
                best[metric_name] = score
                suffix = {
                    "mAP": "map",
                    "G-mIoU@3": "gmiou3",
                    "joint": "joint",
                }[metric_name]
                checkpoint_path = output_dir / f"best_{suffix}.pt"
                save_checkpoint(
                    checkpoint_path, model, optimizer, scheduler,
                    config, args, epoch, metrics,
                )
                shutil.copy2(
                    full_submission_path,
                    output_dir / f"best_{suffix}_val_predictions.jsonl",
                )
                write_json(
                    primary_metrics,
                    output_dir / f"best_{suffix}_val_metrics.json",
                )
                if config.use_hierarchical_counter:
                    for diagnostic_name in ("predictions.jsonl", "metrics.json"):
                        source = output_dir / f"latest_val_{args.diagnostic_mode}_{diagnostic_name}"
                        destination = output_dir / (
                            f"best_{suffix}_val_{args.diagnostic_mode}_{diagnostic_name}"
                        )
                        shutil.copy2(source, destination)
                if metric_name == primary_name:
                    shutil.copy2(checkpoint_path, output_dir / "best.pt")
                    improved_primary = True
            epochs_without_improvement = (
                0 if improved_primary else epochs_without_improvement + 1
            )

        save_checkpoint(
            output_dir / "last.pt", model, optimizer, scheduler, config, args, epoch,
            metrics,
            training_state=make_training_state(best, epochs_without_improvement),
        )
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps(record, ensure_ascii=False))
        if evaluated and args.patience >= 0 and epochs_without_improvement > args.patience:
            print(f"early stop after epoch {epoch + 1}")
            break


if __name__ == "__main__":
    main()
