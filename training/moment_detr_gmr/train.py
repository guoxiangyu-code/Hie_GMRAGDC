from __future__ import annotations

import argparse
import logging
import math
import os
import pprint
import random
import shutil
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from easydict import EasyDict
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    from .config import BaseOptions
    from .dataset import StartEndDataset, prepare_batch_inputs, start_end_collate
    from .evaluate import eval_epoch, setup_model
except ImportError:  # Direct script execution.
    from config import BaseOptions
    from dataset import StartEndDataset, prepare_batch_inputs, start_end_collate
    from evaluate import eval_epoch, setup_model
from models.moment_detr_gmr.utils.basic_utils import (
    AverageMeter,
    rename_latest_to_best,
    save_checkpoint,
    save_json,
    write_log,
)
from models.moment_detr_gmr.utils.model_utils import count_parameters, ModelEMA
from models.moment_detr_gmr.hierarchical_counter import inverse_sqrt_positive_count_weights

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

# Canonical optional-branch structure for every Moment-DETR training variant.
# Tuple order: existence, quality, dual grounding, counter, independent zero,
# learned pairwise selector.
MOMENT_VARIANT_FLAGS = {
    "md_base": (False, False, False, False, False, False),
    "md_gmr": (True, False, False, False, False, False),
    "md_gmr_clean": (True, False, False, False, False, False),
    "md_quality": (True, True, False, False, False, False),
    "md_dual": (True, False, True, False, False, False),
    "md_quality_dual": (True, True, True, False, False, False),
    "md_counter": (True, False, False, True, False, False),
    "md_hiea2m": (True, True, True, True, False, False),
    "md_hiea2m_zero": (True, True, True, True, True, False),
    "md_hiea2m_pairwise": (True, True, True, True, True, True),
}


def set_seed(seed, use_cuda=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)


def gradient_audit(model):
    """Summarize first-batch gradient flow for shared and optional branches."""
    prefixes = (
        "input_txt_proj", "input_vid_proj", "transformer.encoder", "transformer.decoder",
        "query_embed", "class_embed", "span_embed", "exist_head", "quality_embed",
        "dual_grounding", "hierarchical_counter", "zero_verifier_head",
        "pairwise_same_event_head",
    )
    audit = {}
    for prefix in prefixes:
        squared = 0.0
        parameters = 0
        with_gradient = 0
        for name, parameter in model.named_parameters():
            if name != prefix and not name.startswith(prefix + "."):
                continue
            parameters += parameter.numel()
            if parameter.grad is None:
                continue
            with_gradient += parameter.numel()
            squared += float(parameter.grad.detach().float().pow(2).sum())
        if parameters:
            audit[prefix] = {
                "parameters": parameters,
                "parameters_with_grad": with_gradient,
                "gradient_l2": math.sqrt(squared),
            }
    return audit


def save_gradient_audit(model, opt):
    audit = gradient_audit(model)
    required = [
        "input_txt_proj", "input_vid_proj", "transformer.encoder", "transformer.decoder",
        "query_embed", "class_embed", "span_embed",
    ]
    if bool(getattr(opt, "use_exist_head", False)):
        required.append("exist_head")
    if bool(getattr(opt, "use_quality_head", False)):
        required.append("quality_embed")
    if bool(getattr(opt, "use_dual_grounding", False)):
        required.append("dual_grounding")
    if bool(getattr(opt, "use_hierarchical_counter", False)):
        required.append("hierarchical_counter")
    if bool(getattr(opt, "use_independent_zero_head", False)):
        required.append("zero_verifier_head")
    if bool(getattr(opt, "use_pairwise_head", False)):
        required.append("pairwise_same_event_head")
    # Head-only staged training intentionally freezes all other modules.
    required = [
        prefix for prefix in required
        if any(
            parameter.requires_grad
            for name, parameter in model.named_parameters()
            if name == prefix or name.startswith(prefix + ".")
        )
    ]
    broken = []
    for name in required:
        norm = float(audit.get(name, {}).get("gradient_l2", 0.0))
        if not math.isfinite(norm) or norm <= 0:
            broken.append(name)
    if broken:
        raise RuntimeError(f"Disconnected/non-finite first-batch gradients: {broken}")
    save_json(
        {"variant": opt.variant, "groups": audit},
        os.path.join(opt.results_dir, "gradient_audit.json"),
        save_pretty=True,
        sort_keys=True,
    )


def _joint_metric(map_score, gmiou_score, opt):
    """Return a reference-normalized joint score for checkpoint comparison."""
    reference_map = float(getattr(opt, "reference_map", 0.0))
    reference_gmiou = float(getattr(opt, "reference_gmiou3", 0.0))
    if reference_map > 0 and reference_gmiou > 0:
        # A gain on one objective cannot conceal a regression on the other.
        return min(map_score / reference_map, gmiou_score / reference_gmiou)
    return 2 * map_score * gmiou_score / max(map_score + gmiou_score, 1e-8)


def _save_named_checkpoint(model, optimizer, lr_scheduler, epoch_i, opt, filename):
    """Save an objective-specific checkpoint without changing the main path."""
    original_path = opt.ckpt_filepath
    try:
        opt.ckpt_filepath = os.path.join(opt.results_dir, filename)
        save_checkpoint(model, optimizer, lr_scheduler, epoch_i, opt)
    finally:
        opt.ckpt_filepath = original_path


def _copy_latest_artifacts(latest_file_paths, objective):
    """Snapshot predictions/metrics for an objective-specific best epoch."""
    copied_paths = []
    for source in latest_file_paths:
        basename = os.path.basename(source)
        if basename.startswith("latest_"):
            basename = f"best_{objective}_" + basename[len("latest_"):]
        else:
            stem, suffix = os.path.splitext(basename)
            basename = f"{stem}_best_{objective}{suffix}"
        destination = os.path.join(os.path.dirname(source), basename)
        shutil.copy2(source, destination)
        copied_paths.append(destination)
    return copied_paths

def train_epoch(model, criterion, train_loader, optimizer, opt, epoch_i, model_ema=None):
    logger.info("[Epoch %d]", epoch_i + 1)
    model.train()
    trainable_scope = str(getattr(opt, "trainable_scope", "all"))
    if trainable_scope != "all":
        # Frozen feature producers must also have deterministic dropout/normalization
        # behavior; only the staged head(s) stay in training mode.
        model.eval()
        if trainable_scope in {"zero", "selection_heads"} \
                and getattr(model, "zero_verifier_head", None) is not None:
            model.zero_verifier_head.train()
        if trainable_scope in {"pairwise", "selection_heads"} \
                and getattr(model, "pairwise_same_event_head", None) is not None:
            model.pairwise_same_event_head.train()
        if trainable_scope == "selection_heads" \
                and getattr(model, "hierarchical_counter", None) is not None:
            model.hierarchical_counter.train()
    criterion.train()
    loss_meters = defaultdict(AverageMeter)
    gradient_audit_pending = epoch_i == 0

    for batch_index, batch in enumerate(tqdm(train_loader, desc="Training Iteration")):
        model_inputs, targets = prepare_batch_inputs(batch[1], opt.device)
        outputs = model(**model_inputs)
        loss_dict = criterion(outputs, targets)
        losses = sum(
            loss_dict[k] * criterion.weight_dict[k]
            for k in loss_dict.keys()
            if k in criterion.weight_dict
        )

        optimizer.zero_grad()
        losses.backward()
        # Localization heads legitimately have zero gradient on an all-null
        # batch. Audit the first batch containing at least one ground-truth
        # moment so zero-gradient checks remain meaningful.
        has_positive = bool(targets["exist_label"].sum().detach().item() > 0)
        if gradient_audit_pending and has_positive:
            save_gradient_audit(model, opt)
            gradient_audit_pending = False
        if opt.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
        optimizer.step()
        if model_ema is not None:
            model_ema.update(model)

        loss_dict["loss_overall"] = float(losses.detach())
        for k, v in loss_dict.items():
            scalar = float(v.detach()) if torch.is_tensor(v) else float(v)
            loss_meters[k].update(
                scalar * criterion.weight_dict[k] if k in criterion.weight_dict else scalar
            )

    if gradient_audit_pending:
        raise RuntimeError("No positive sample was available for the first-epoch gradient audit")

    write_log(opt, epoch_i, loss_meters)

def train(model, criterion, optimizer, lr_scheduler, train_dataset, val_dataset, opt):
    opt.train_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str}\n"
    opt.eval_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str} [Metrics] {eval_metrics_str}\n"

    if str(getattr(opt, "selection_metric", "mAP")) == "joint" and (
        float(getattr(opt, "reference_map", 0.0)) <= 0
        or float(getattr(opt, "reference_gmiou3", 0.0)) <= 0
    ):
        raise ValueError(
            "selection_metric='joint' requires positive --reference_map and "
            "--reference_gmiou3 from a matched full-validation baseline"
        )

    train_loader = DataLoader(
        train_dataset,
        collate_fn=start_end_collate,
        batch_size=opt.bsz,
        num_workers=opt.num_workers,
        shuffle=True,
    )

    model_ema = None
    if opt.model_ema:
        logger.info("Using model EMA")
        model_ema = ModelEMA(model, decay=opt.ema_decay)

    prev_best_score = float("-inf")
    best_by_objective = {
        "map": {"score": float("-inf"), "epoch": None},
        "gmiou3": {"score": float("-inf"), "epoch": None},
        "joint": {"score": float("-inf"), "epoch": None},
    }
    es_cnt = 0
    save_submission_filename = f"latest_{opt.dset_name}_val_preds.jsonl"

    for epoch_i in trange(opt.n_epoch, desc="Epoch"):
        train_epoch(model, criterion, train_loader, optimizer, opt, epoch_i, model_ema)
        lr_scheduler.step()

        if (epoch_i + 1) % opt.eval_epoch_interval != 0:
            continue

        with torch.no_grad():
            eval_model = model_ema.module if model_ema is not None else model
            metrics, eval_loss_meters, latest_file_paths = eval_epoch(
                epoch_i,
                eval_model,
                val_dataset,
                opt,
                save_submission_filename,
                criterion,
            )

        write_log(opt, epoch_i, eval_loss_meters, metrics=metrics, mode="val")
        logger.info("metrics %s", pprint.pformat(metrics["brief"], indent=4))
        map_score = float(metrics["brief"].get("mAP", 0.0))
        gmiou_score = float(metrics["brief"].get("G-mIoU@3", 0.0))
        joint_score = _joint_metric(map_score, gmiou_score, opt)
        objective_scores = {
            "map": map_score,
            "gmiou3": gmiou_score,
            "joint": joint_score,
        }
        for objective, score in objective_scores.items():
            if score <= best_by_objective[objective]["score"]:
                continue
            best_by_objective[objective] = {
                "score": score,
                "epoch": epoch_i + 1,
                "mAP": map_score,
                "G-mIoU@3": gmiou_score,
            }
            _save_named_checkpoint(
                eval_model,
                optimizer,
                lr_scheduler,
                epoch_i,
                opt,
                f"best_{objective}.ckpt",
            )
            _copy_latest_artifacts(latest_file_paths, objective)
            logger.info(
                "Updated best_%s checkpoint: score=%.4f mAP=%.4f G-mIoU@3=%.4f",
                objective,
                score,
                map_score,
                gmiou_score,
            )
        save_json(
            best_by_objective,
            os.path.join(opt.results_dir, "best_objectives.json"),
            save_pretty=True,
            sort_keys=True,
        )

        selection_metric = str(getattr(opt, "selection_metric", "mAP"))
        if selection_metric == "mAP":
            stop_score = map_score
        elif selection_metric == "gmiou3":
            stop_score = gmiou_score
        elif selection_metric == "joint":
            stop_score = joint_score
        else:
            raise ValueError(f"Unknown selection_metric={selection_metric!r}")

        if stop_score > prev_best_score:
            prev_best_score = stop_score
            save_checkpoint(eval_model, optimizer, lr_scheduler, epoch_i, opt)
            rename_latest_to_best(latest_file_paths)
            es_cnt = 0
            logger.info("Updated best checkpoint.")
        else:
            es_cnt += 1
            logger.info("Early stop counter: %d/%d", es_cnt, opt.max_es_cnt)
            if es_cnt >= int(opt.max_es_cnt):
                logger.info("Early stopping at epoch %d. Best score %.4f", epoch_i + 1, prev_best_score)
                break

def build_dataset_config(opt, data_path, load_labels=True, keep_empty_gt=False):
    return EasyDict(
        dset_name=opt.dset_name,
        domain=None,
        data_path=data_path,
        ctx_mode=opt.ctx_mode,
        v_feat_dirs=opt.v_feat_dirs,
        a_feat_dirs=None,
        q_feat_dir=opt.t_feat_dir,
        q_feat_type="last_hidden_state",
        v_feat_types=opt.v_feat_types,
        a_feat_types=None,
        max_q_l=opt.max_q_l,
        max_v_l=opt.max_v_l,
        max_a_l=opt.max_a_l,
        clip_len=opt.clip_length,
        max_windows=opt.max_windows,
        span_loss_type=opt.span_loss_type,
        load_labels=load_labels,
        mr_only=bool(getattr(opt, "mr_only", True)),
        keep_empty_gt=keep_empty_gt,
        trim_text_by_attention_mask=bool(
            getattr(opt, "trim_text_by_attention_mask", False)
        ),
    )

def main(opt, resume=None):
    logger.info("Setup config, data and model...")
    set_seed(opt.seed, use_cuda=str(opt.device).startswith("cuda"))

    train_dataset = StartEndDataset(**build_dataset_config(
        opt,
        opt.train_path,
        load_labels=True,
        keep_empty_gt=(
            bool(getattr(opt, "use_exist_head", False))
            or bool(getattr(opt, "use_hierarchical_counter", False))
        ),
    ))
    val_dataset = StartEndDataset(**build_dataset_config(
        opt,
        opt.eval_path,
        load_labels=True,
        keep_empty_gt=True,
    ))

    expected_train = sum(1 for _ in open(opt.train_path, "r", encoding="utf-8"))
    expected_val = sum(1 for _ in open(opt.eval_path, "r", encoding="utf-8"))
    expected_train = expected_train if (
        bool(getattr(opt, "use_exist_head", False))
        or bool(getattr(opt, "use_hierarchical_counter", False))
    ) else sum(1 for row in train_dataset.data)
    if len(train_dataset) != expected_train or len(val_dataset) != expected_val:
        raise RuntimeError(
            f"Incomplete feature coverage: train={len(train_dataset)}/{expected_train}, "
            f"val={len(val_dataset)}/{expected_val}"
        )

    if bool(getattr(opt, "use_hierarchical_counter", False)):
        positive_counts = [0, 0, 0, 0]
        for row in train_dataset.data:
            count = min(len(row.get("relevant_windows") or []), 4)
            if count > 0:
                positive_counts[count - 1] += 1
        opt.positive_count_weights = inverse_sqrt_positive_count_weights(
            positive_counts
        ).tolist()
        logger.info(
            "Positive count classes=%s weights=%s",
            positive_counts,
            opt.positive_count_weights,
        )

    model, criterion, optimizer, lr_scheduler = setup_model(opt)
    if resume is not None:
        checkpoint = torch.load(resume, map_location="cpu", weights_only=False)
        checkpoint_state = checkpoint.get("model", checkpoint)
        saved_opt = checkpoint.get("opt") if isinstance(checkpoint, dict) else None
        incompatible = model.load_state_dict(checkpoint_state, strict=False)
        allowed_missing_prefixes = []
        if bool(getattr(opt, "use_quality_head", False)):
            allowed_missing_prefixes.append("quality_embed.")
        if bool(getattr(opt, "use_dual_grounding", False)):
            allowed_missing_prefixes.append("dual_grounding.")
        if bool(getattr(opt, "use_hierarchical_counter", False)):
            allowed_missing_prefixes.append("hierarchical_counter.")
        if bool(getattr(opt, "use_independent_zero_head", False)):
            allowed_missing_prefixes.append("zero_verifier_head.")
        if bool(getattr(opt, "use_pairwise_head", False)):
            allowed_missing_prefixes.append("pairwise_same_event_head.")
        if bool(getattr(opt, "use_exist_head", False)):
            allowed_missing_prefixes.append("exist_head.")
        # A whole optional module may be new when migrating a parent model;
        # a partially present module instead indicates a corrupt/incompatible
        # checkpoint and must not be silently reinitialized.
        allowed_missing_prefixes = [
            prefix for prefix in allowed_missing_prefixes
            if not any(key.startswith(prefix) for key in checkpoint_state)
        ]
        invalid_missing = [
            key for key in incompatible.missing_keys
            if not any(key.startswith(prefix) for prefix in allowed_missing_prefixes)
        ]
        allowed_unexpected_prefixes = ["exist_head."] if not opt.use_exist_head else []
        invalid_unexpected = [
            key for key in incompatible.unexpected_keys
            if not any(key.startswith(prefix) for prefix in allowed_unexpected_prefixes)
        ]
        if invalid_missing or invalid_unexpected:
            raise RuntimeError(
                "Unsafe checkpoint migration: "
                f"missing={invalid_missing}, unexpected={invalid_unexpected}"
            )
        run_protocol = {
            "trim_text_by_attention_mask": bool(opt.trim_text_by_attention_mask),
            "round_to_clip": bool(opt.round_to_clip),
            "mask_null_vmr_loss": bool(
                getattr(opt, "mask_null_vmr_loss", False)
            ),
            "max_q_l": int(opt.max_q_l),
            "max_windows": int(opt.max_windows),
        }
        checkpoint_protocol = {
            # Older release checkpoints predate these explicit switches.
            "trim_text_by_attention_mask": bool(
                getattr(saved_opt, "trim_text_by_attention_mask", False)
            ),
            "round_to_clip": bool(getattr(saved_opt, "round_to_clip", True)),
            "mask_null_vmr_loss": bool(
                getattr(saved_opt, "mask_null_vmr_loss", False)
            ),
            "max_q_l": int(getattr(saved_opt, "max_q_l", opt.max_q_l)),
            "max_windows": int(getattr(saved_opt, "max_windows", opt.max_windows)),
        }
        protocol_match = run_protocol == checkpoint_protocol
        if not protocol_match:
            logger.warning(
                "Warm-start protocol differs from checkpoint: checkpoint=%s run=%s. "
                "Treat this as a protocol-transfer run, not a step-zero matched ablation.",
                checkpoint_protocol,
                run_protocol,
            )
        save_json(
            {
                "resume": os.path.abspath(resume),
                "checkpoint_protocol": checkpoint_protocol,
                "run_protocol": run_protocol,
                "protocol_match": protocol_match,
                "initialized_new_keys": list(incompatible.missing_keys),
                "discarded_checkpoint_keys": list(incompatible.unexpected_keys),
            },
            os.path.join(opt.results_dir, "initialization_audit.json"),
            save_pretty=True,
            sort_keys=True,
        )
        logger.info("Loaded model checkpoint: %s", resume)

    count_parameters(model)
    logger.info("Start training")
    train(model, criterion, optimizer, lr_scheduler, train_dataset, val_dataset, opt)

def parse_args():
    parser = argparse.ArgumentParser(description="Train Moment-DETR-GMR on Soccer-GMR features.")
    parser.add_argument("--model", "-m", default="moment_detr", choices=["moment_detr"])
    parser.add_argument("--dataset", "-d", default="soccer_gmr", choices=["soccer_gmr"])
    parser.add_argument("--feature", "-f", default="clip_slowfast", choices=["clip_slowfast"])
    parser.add_argument("--resume", "-r", type=str, default=None, help="Optional checkpoint for fine-tuning.")
    parser.add_argument(
        "--variant",
        choices=tuple(MOMENT_VARIANT_FLAGS),
        default="md_gmr",
    )
    parser.add_argument("--run_tag", type=str, default=None, help="Append a tag to the output directory.")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--n_epoch", type=int, default=None)
    parser.add_argument("--bsz", type=int, default=None)
    parser.add_argument("--eval_bsz", type=int, default=None)
    parser.add_argument("--max_es_cnt", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--eval_epoch_interval", type=int, default=None)
    parser.add_argument("--exist_gate_thd", type=float, default=None)
    parser.add_argument("--gmiou_cls_threshold", type=float, default=None)
    parser.add_argument("--quality_score_alpha", type=float, default=None)
    parser.add_argument("--diversity_lambda", type=float, default=None)
    parser.add_argument("--count_exist_thd", type=float, default=None)
    parser.add_argument(
        "--exist_score_mode", choices=["gate", "zero", "rescue", "cascade"], default=None
    )
    parser.add_argument("--gate_recall_thd", type=float, default=None)
    parser.add_argument("--zero_decision_thd", type=float, default=None)
    parser.add_argument("--zero_veto_thd", type=float, default=None)
    parser.add_argument("--zero_localization_thd", type=float, default=None)
    parser.add_argument(
        "--set_selection_mode",
        choices=[
            "legacy", "direct_topk", "learned_topk",
            "learned_soft_count", "learned_soft_count_fusion",
        ],
        default=None,
    )
    parser.add_argument("--selection_k", type=int, default=None)
    parser.add_argument("--selection_max_output", type=int, default=None)
    parser.add_argument("--pairwise_redundancy_lambda", type=float, default=None)
    parser.add_argument("--count_prior_weight", type=float, default=None)
    parser.add_argument("--selection_stop_thd", type=float, default=None)
    parser.add_argument("--same_event_thd", type=float, default=None)
    parser.add_argument("--fusion_boundary_std_thd", type=float, default=None)
    parser.add_argument("--zero_positive_query_weight", type=float, default=None)
    parser.add_argument("--pair_assignment_iou", type=float, default=None)
    parser.add_argument("--pair_ambiguity_margin", type=float, default=None)
    parser.add_argument("--pair_hard_negative_weight", type=float, default=None)
    parser.add_argument(
        "--trainable_scope",
        choices=["all", "zero", "pairwise", "selection_heads"],
        default=None,
    )
    parser.add_argument(
        "--decode_mode", choices=["full", "threshold", "adaptive", "hard"], default=None
    )
    parser.add_argument("--selection_metric", choices=["mAP", "gmiou3", "joint"], default=None)
    parser.add_argument("--reference_map", type=float, default=None)
    parser.add_argument("--reference_gmiou3", type=float, default=None)
    parser.add_argument(
        "--trim_text_by_attention_mask", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--round_to_clip", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--mask-null-vmr-loss",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "apply the GMR indicator I(y=1) to localization/query-quality "
            "losses so null queries supervise only GMR decision heads"
        ),
    )
    parser.add_argument("--train_path", type=str, default=None)
    parser.add_argument("--eval_path", type=str, default=None)
    parser.add_argument("--t_feat_dir", type=str, default=None)
    parser.add_argument("--v_feat_dirs", type=str, nargs="+", default=None)
    parser.add_argument("--results_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true", help="Remove the output directory before training.")
    parser.add_argument("--mr_only", action="store_true", default=True, help="Disable saliency labels.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    option_manager = BaseOptions(args.model, args.dataset, args.feature, args.resume)
    option_manager.parse()
    opt = option_manager.option

    if args.run_tag:
        opt.results_dir = os.path.join(opt.results_dir, args.run_tag)
        opt.ckpt_filepath = os.path.join(opt.results_dir, opt.ckpt_filename)
        opt.train_log_filepath = os.path.join(opt.results_dir, opt.train_log_filename)
        opt.eval_log_filepath = os.path.join(opt.results_dir, opt.eval_log_filename)
    for name in [
        "seed", "lr", "n_epoch", "bsz", "eval_bsz", "max_es_cnt", "num_workers",
        "eval_epoch_interval", "exist_gate_thd", "gmiou_cls_threshold",
        "quality_score_alpha", "diversity_lambda", "count_exist_thd", "decode_mode", "selection_metric",
        "zero_positive_query_weight", "pair_assignment_iou",
        "pair_ambiguity_margin", "pair_hard_negative_weight", "trainable_scope",
        "exist_score_mode", "gate_recall_thd", "zero_decision_thd",
        "zero_veto_thd", "zero_localization_thd",
        "set_selection_mode", "selection_k", "selection_max_output",
        "pairwise_redundancy_lambda", "count_prior_weight", "selection_stop_thd",
        "same_event_thd", "fusion_boundary_std_thd",
        "reference_map", "reference_gmiou3", "trim_text_by_attention_mask",
        "round_to_clip", "mask_null_vmr_loss", "train_path", "eval_path",
        "t_feat_dir", "results_dir", "device",
    ]:
        value = getattr(args, name)
        if value is not None:
            setattr(opt, name, value)
    if args.v_feat_dirs is not None:
        opt.v_feat_dirs = args.v_feat_dirs
    if args.results_dir is not None:
        opt.ckpt_filepath = os.path.join(opt.results_dir, opt.ckpt_filename)
        opt.train_log_filepath = os.path.join(opt.results_dir, opt.train_log_filename)
        opt.eval_log_filepath = os.path.join(opt.results_dir, opt.eval_log_filename)
    opt.mr_only = True
    opt.lw_saliency = 0
    opt.variant = args.variant
    (
        opt.use_exist_head,
        opt.use_quality_head,
        opt.use_dual_grounding,
        opt.use_hierarchical_counter,
        opt.use_independent_zero_head,
        opt.use_pairwise_head,
    ) = MOMENT_VARIANT_FLAGS[args.variant]
    if args.variant not in {"md_base", "md_gmr"}:
        if args.trim_text_by_attention_mask is None:
            opt.trim_text_by_attention_mask = True
        if args.round_to_clip is None:
            opt.round_to_clip = False
    if opt.use_independent_zero_head and args.mask_null_vmr_loss is None:
        opt.mask_null_vmr_loss = True
    if args.trainable_scope is None:
        opt.trainable_scope = "all"
    # Training/early stopping always defaults to the full DETR set. The
    # HieA2G-style adaptive count decoder is calibrated and evaluated as a
    # separate validation view so an immature count head cannot erase useful
    # localization proposals.
    if args.decode_mode is None:
        opt.decode_mode = "full"

    option_manager.clean_and_makedirs(overwrite=args.overwrite)
    main(opt, resume=args.resume)
