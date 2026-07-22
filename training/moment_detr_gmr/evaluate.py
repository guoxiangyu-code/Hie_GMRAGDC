from __future__ import annotations

import argparse
import logging
import os
import pprint
import sys
from collections import defaultdict

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from easydict import EasyDict
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    from .config import BaseOptions
    from .dataset import StartEndDataset, prepare_batch_inputs, start_end_collate
    from .postprocessing import PostProcessorDETR
except ImportError:  # Direct script execution.
    from config import BaseOptions
    from dataset import StartEndDataset, prepare_batch_inputs, start_end_collate
    from postprocessing import PostProcessorDETR
from models.moment_detr_gmr.moment_detr import build_model as build_model_moment_detr
from models.moment_detr_gmr.hierarchical_counter import hierarchical_count_probabilities
from models.moment_detr_gmr.set_decoder import (
    adaptive_count_indices,
    diversity_ranking,
    fuse_query_scores,
)
from models.moment_detr_gmr.learned_selector import (
    cautious_complete_link_fusion,
    combine_two_stage_existence,
    learned_mmr_select,
    two_stage_accept,
)
from models.moment_detr_gmr.utils.basic_utils import AverageMeter, save_json, save_jsonl
from models.moment_detr_gmr.utils.span_utils import span_cxw_to_xx
from eval.eval_main import evaluate_gmr

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

def eval_epoch_post_processing(submission, opt, gt_data, save_submission_filename):
    logger.info("Saving/evaluating predictions")
    submission_path = os.path.join(opt.results_dir, save_submission_filename)
    save_jsonl(submission, submission_path)

    if opt.eval_split_name == "val":
        metrics = evaluate_gmr(
            submission,
            gt_data,
            k_list=tuple(getattr(opt, "eval_k_list", (1, 3, 5))),
            max_pred_windows=int(getattr(opt, "max_pred_windows", 10)),
            cls_thresholds=tuple(getattr(opt, "cls_thresholds", (0.4, 0.6, 0.8))),
            gmiou_cls_threshold=float(getattr(opt, "gmiou_cls_threshold", 0.4)),
            map_num_workers=int(getattr(opt, "map_num_workers", 8)),
            verbose=False,
        )
        save_metrics_path = submission_path.replace(".jsonl", "_metrics.json")
        save_json(metrics, save_metrics_path, save_pretty=True, sort_keys=False)
        latest_file_paths = [submission_path, save_metrics_path]
    else:
        metrics = None
        latest_file_paths = [submission_path]

    return metrics, latest_file_paths

@torch.no_grad()
def compute_mr_results(epoch_i, model, eval_loader, opt, criterion=None):
    del epoch_i
    loss_meters = defaultdict(AverageMeter)
    mr_res = []

    for batch in tqdm(eval_loader, desc="compute moment scores"):
        query_meta = batch[0]
        model_inputs, targets = prepare_batch_inputs(batch[1], opt.device)
        outputs = model(**model_inputs)

        pred_spans = outputs["pred_spans"].cpu()
        prob = F.softmax(outputs["pred_logits"], -1)
        foreground_scores = prob[..., 0].detach().cpu()
        quality_probabilities = (
            torch.sigmoid(outputs["pred_quality_logits"]).detach().cpu()
            if "pred_quality_logits" in outputs else None
        )
        scores = fuse_query_scores(
            prob[..., 0],
            outputs.get("pred_quality_logits"),
            quality_alpha=float(getattr(opt, "quality_score_alpha", 0.5)),
        ).detach().cpu()

        pred_exist_scores = None
        pred_gate_scores = None
        pred_zero_scores = None
        pred_exist_decisions = None
        localization_evidence = scores.max(dim=1).values
        if "pred_exist_logits" in outputs:
            pred_gate_scores = torch.sigmoid(
                outputs.get("pred_gate_logits", outputs["pred_exist_logits"])
            ).detach().cpu()
            pred_exist_scores = pred_gate_scores
        if "pred_zero_logits" in outputs:
            pred_zero_scores = torch.sigmoid(outputs["pred_zero_logits"]).detach().cpu()
            pred_exist_scores = combine_two_stage_existence(
                pred_gate_scores,
                pred_zero_scores,
                localization_evidence,
                mode=str(getattr(opt, "exist_score_mode", "cascade")),
                veto_threshold=float(getattr(opt, "zero_veto_thd", 0.7)),
                localization_threshold=float(
                    getattr(opt, "zero_localization_thd", 0.2)
                ),
            )
            pred_exist_decisions = two_stage_accept(
                pred_gate_scores,
                pred_zero_scores,
                localization_evidence,
                gate_threshold=float(getattr(opt, "gate_recall_thd", 0.3)),
                zero_threshold=float(getattr(opt, "zero_decision_thd", 0.6)),
                veto_threshold=float(getattr(opt, "zero_veto_thd", 0.7)),
                localization_threshold=float(
                    getattr(opt, "zero_localization_thd", 0.2)
                ),
            )

        same_event_probabilities = (
            torch.sigmoid(outputs["pred_same_event_logits"]).detach().cpu()
            if "pred_same_event_logits" in outputs else None
        )

        count_probabilities = None
        if "pred_positive_count_logits" in outputs:
            count_probabilities = hierarchical_count_probabilities(outputs).detach().cpu()

        for idx, (meta, spans, score) in enumerate(zip(query_meta, pred_spans, scores)):
            spans = span_cxw_to_xx(spans).clamp(0, 1)
            sample_count_probabilities = (
                count_probabilities[idx] if count_probabilities is not None else None
            )
            selection_mode = str(getattr(opt, "set_selection_mode", "legacy"))
            selected_spans = None
            selected_scores = None
            if selection_mode == "legacy":
                ranking = diversity_ranking(
                    spans,
                    score,
                    diversity_lambda=float(getattr(opt, "diversity_lambda", 0.0)),
                )
                selected_indices = adaptive_count_indices(
                    ranking,
                    score,
                    sample_count_probabilities,
                    mode=str(getattr(opt, "decode_mode", "full")),
                    existence_threshold=float(getattr(opt, "count_exist_thd", 0.4)),
                    count_confidence_threshold=float(
                        getattr(opt, "count_confidence_thd", 0.55)
                    ),
                    window_score_threshold=float(getattr(opt, "window_score_thd", 0.1)),
                )
            elif selection_mode == "direct_topk":
                selected_indices = torch.argsort(score, descending=True)[
                    : int(getattr(opt, "selection_k", 3))
                ].tolist()
                ranking = torch.argsort(score, descending=True).tolist()
            elif selection_mode in {
                "learned_topk", "learned_soft_count", "learned_soft_count_fusion"
            }:
                if same_event_probabilities is None:
                    raise RuntimeError(
                        f"set_selection_mode={selection_mode!r} requires a pairwise head"
                    )
                fixed_topk = selection_mode == "learned_topk"
                selection = learned_mmr_select(
                    score,
                    same_event_probabilities[idx],
                    max_output=(
                        int(getattr(opt, "selection_k", 3)) if fixed_topk
                        else int(getattr(opt, "selection_max_output", 10))
                    ),
                    redundancy_lambda=float(
                        getattr(opt, "pairwise_redundancy_lambda", 1.0)
                    ),
                    count_probabilities=(
                        None if fixed_topk else sample_count_probabilities
                    ),
                    count_prior_weight=(
                        0.0 if fixed_topk else float(
                            getattr(opt, "count_prior_weight", 0.5)
                        )
                    ),
                    stop_threshold=(
                        float("-inf") if fixed_topk else float(
                            getattr(opt, "selection_stop_thd", -1.0)
                        )
                    ),
                )
                selected_indices = selection.selected
                ranking = selected_indices + [
                    index for index in torch.argsort(score, descending=True).tolist()
                    if index not in set(selected_indices)
                ]
                if selection_mode == "learned_soft_count_fusion":
                    selected_spans, selected_scores = cautious_complete_link_fusion(
                        spans,
                        score,
                        same_event_probabilities[idx],
                        selected_indices,
                        same_event_threshold=float(
                            getattr(opt, "same_event_thd", 0.8)
                        ),
                        boundary_std_threshold=float(
                            getattr(opt, "fusion_boundary_std_thd", 0.03)
                        ),
                    )
            else:
                raise ValueError(f"Unknown set_selection_mode={selection_mode!r}")
            spans_seconds = spans * float(meta["duration"])
            all_windows = torch.cat([spans_seconds, score[:, None]], dim=1).tolist()
            if selected_spans is None:
                cur_ranked_preds = [all_windows[index] for index in selected_indices]
            else:
                fused_seconds = selected_spans * float(meta["duration"])
                cur_ranked_preds = torch.cat(
                    [fused_seconds, selected_scores[:, None]], dim=1
                ).tolist()
            cur_ranked_preds = [[float(f"{e:.4f}") for e in row] for row in cur_ranked_preds]

            cur_query_pred = {
                "qid": meta["qid"],
                "query": meta["query"],
                "vid": meta["vid"],
                "pred_relevant_windows": cur_ranked_preds,
            }
            if pred_exist_scores is not None:
                cur_query_pred["pred_exist_score"] = float(f"{float(pred_exist_scores[idx]):.4f}")
            if pred_gate_scores is not None:
                cur_query_pred["pred_gate_score"] = float(
                    f"{float(pred_gate_scores[idx]):.6f}"
                )
            if pred_zero_scores is not None:
                cur_query_pred["pred_zero_score"] = float(
                    f"{float(pred_zero_scores[idx]):.6f}"
                )
                cur_query_pred["pred_localization_evidence"] = float(
                    f"{float(localization_evidence[idx]):.6f}"
                )
                cur_query_pred["pred_exist_decision"] = int(
                    bool(pred_exist_decisions[idx])
                )
            if sample_count_probabilities is not None:
                predicted_positive_count = int(
                    torch.argmax(sample_count_probabilities[1:]).item()
                ) + 1
                predicted_exists = (
                    1.0 - float(sample_count_probabilities[0])
                    > float(getattr(opt, "count_exist_thd", 0.4))
                )
                cur_query_pred["pred_count"] = (
                    predicted_positive_count if predicted_exists else 0
                )
                cur_query_pred["pred_count_probs"] = [
                    float(f"{float(value):.6f}") for value in sample_count_probabilities
                ]
            if bool(getattr(opt, "save_raw_queries", False)):
                cur_query_pred["all_query_windows"] = [
                    [float(f"{value:.4f}") for value in all_windows[index]]
                    for index in ranking
                ]
                cur_query_pred["all_query_indices"] = list(ranking)
                raw_quality = (
                    quality_probabilities[idx]
                    if quality_probabilities is not None
                    else torch.ones_like(foreground_scores[idx])
                )
                cur_query_pred["all_query_components"] = [
                    [
                        float(f"{float(spans_seconds[index, 0]):.4f}"),
                        float(f"{float(spans_seconds[index, 1]):.4f}"),
                        float(f"{float(foreground_scores[idx, index]):.6f}"),
                        float(f"{float(raw_quality[index]):.6f}"),
                    ]
                    for index in range(len(all_windows))
                ]
                if same_event_probabilities is not None:
                    ranked_pairwise = same_event_probabilities[idx][ranking][:, ranking]
                    cur_query_pred["pred_same_event_probs"] = [
                        [float(f"{float(value):.6f}") for value in pair]
                        for pair in ranked_pairwise
                    ]
            mr_res.append(cur_query_pred)

        if criterion is not None:
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            loss_dict["loss_overall"] = float(losses)
            for k, v in loss_dict.items():
                loss_meters[k].update(float(v) * weight_dict[k] if k in weight_dict else float(v))

    if bool(getattr(opt, "round_to_clip", True)):
        post_processor = PostProcessorDETR(
            clip_length=opt.clip_length,
            min_ts_val=0,
            max_ts_val=float(getattr(opt, "max_ts_val", 150)),
            min_w_l=1,
            max_w_l=float(getattr(opt, "max_ts_val", 150)),
            move_window_method="left",
            process_func_names=("clip_ts", "round_multiple"),
        )
        mr_res = post_processor(mr_res)
    return mr_res, loss_meters

def eval_epoch(epoch_i, model, eval_dataset, opt, save_submission_filename, criterion=None):
    logger.info("Generate submissions")
    model.eval()
    if criterion is not None:
        criterion.eval()

    eval_loader = DataLoader(
        eval_dataset,
        collate_fn=start_end_collate,
        batch_size=opt.eval_bsz,
        num_workers=opt.num_workers,
        shuffle=False,
    )

    submission, eval_loss_meters = compute_mr_results(epoch_i, model, eval_loader, opt, criterion)
    metrics, latest_file_paths = eval_epoch_post_processing(
        submission,
        opt,
        eval_dataset.data,
        save_submission_filename,
    )
    return metrics, eval_loss_meters, latest_file_paths

def build_model(opt):
    return build_model_moment_detr(opt)

def setup_model(opt):
    logger.info("setup model/optimizer/scheduler")
    model, criterion = build_model(opt)

    trainable_scope = str(getattr(opt, "trainable_scope", "all"))
    scope_prefixes = {
        "all": None,
        "zero": ("zero_verifier_head.",),
        "pairwise": ("pairwise_same_event_head.",),
        "selection_heads": (
            "zero_verifier_head.", "pairwise_same_event_head.",
            "hierarchical_counter.",
        ),
    }
    if trainable_scope not in scope_prefixes:
        raise ValueError(f"Unknown trainable_scope={trainable_scope!r}")
    allowed_prefixes = scope_prefixes[trainable_scope]
    if allowed_prefixes is not None:
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith(allowed_prefixes))

    model.to(opt.device)
    criterion.to(opt.device)
    if str(opt.device).startswith("cuda"):
        logger.info("CUDA enabled on %s.", opt.device)

    use_new_modules = any(bool(getattr(opt, name, False)) for name in (
        "use_quality_head", "use_dual_grounding", "use_hierarchical_counter",
        "use_independent_zero_head", "use_pairwise_head",
    ))
    if use_new_modules:
        new_prefixes = (
            "quality_embed.", "dual_grounding.", "hierarchical_counter.",
            "zero_verifier_head.", "pairwise_same_event_head.",
        )
        new_parameters = []
        shared_parameters = []
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            target = new_parameters if name.startswith(new_prefixes) else shared_parameters
            target.append(parameter)
        optimizer_groups = [group for group in [
            {
                "params": shared_parameters,
                "lr": float(opt.lr) * float(getattr(opt, "backbone_lr_scale", 0.1)),
                "name": "shared_backbone",
            },
            {"params": new_parameters, "lr": float(opt.lr), "name": "new_modules"},
        ] if group["params"]]
    else:
        optimizer_groups = [{
            "params": [p for p in model.parameters() if p.requires_grad],
            "lr": float(opt.lr),
            "name": "all",
        }]
    if not optimizer_groups or not any(group["params"] for group in optimizer_groups):
        raise ValueError(f"trainable_scope={trainable_scope!r} selected no parameters")
    optimizer = torch.optim.AdamW(optimizer_groups, lr=opt.lr, weight_decay=opt.wd)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, opt.lr_drop)
    return model, criterion, optimizer, lr_scheduler

def build_dataset_config(opt, data_path, load_labels):
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
        mr_only=True,
        keep_empty_gt=True,
        trim_text_by_attention_mask=bool(
            getattr(opt, "trim_text_by_attention_mask", False)
        ),
    )

def start_inference(opt):
    logger.info("Setup config, data and model...")
    cudnn.benchmark = True
    cudnn.deterministic = False

    checkpoint = torch.load(opt.model_path, map_location="cpu", weights_only=False)
    ckpt_state = checkpoint.get("model")
    if ckpt_state is None:
        raise ValueError(f"Checkpoint missing key 'model': {opt.model_path}")

    has_exist_head = any(k.startswith("exist_head.") for k in ckpt_state.keys())
    has_dual_grounding = any("dual_grounding" in k for k in ckpt_state.keys())
    has_hierarchical_counter = any("hierarchical_counter" in k for k in ckpt_state.keys())
    has_quality_head = any("quality_embed" in k for k in ckpt_state.keys())
    has_independent_zero_head = any(
        k.startswith("zero_verifier_head.") for k in ckpt_state.keys()
    )
    has_pairwise_head = any(
        k.startswith("pairwise_same_event_head.") for k in ckpt_state.keys()
    )
    if bool(getattr(opt, "use_exist_head", False)) != has_exist_head:
        logger.warning(
            "Config/checkpoint mismatch for existence head. config.use_exist_head=%s, ckpt_has_exist_head=%s. "
            "Using checkpoint setting.",
            bool(getattr(opt, "use_exist_head", False)),
            has_exist_head,
        )
        opt.use_exist_head = has_exist_head
    opt.use_dual_grounding = has_dual_grounding
    opt.use_hierarchical_counter = has_hierarchical_counter
    opt.use_quality_head = has_quality_head
    opt.use_independent_zero_head = has_independent_zero_head
    opt.use_pairwise_head = has_pairwise_head
    if checkpoint.get("opt") is not None:
        saved_opt = checkpoint["opt"]
        for name in (
            "max_q_l", "max_v_l", "max_windows", "clip_length", "exist_pool",
            "trim_text_by_attention_mask", "quality_score_alpha", "dual_num_phrases",
            "dual_num_dummies", "dual_slot_iterations", "dual_gate_init", "dual_nheads",
            "counter_dropout", "counter_detach_scores", "decode_mode", "diversity_lambda",
            "count_exist_thd", "count_confidence_thd", "window_score_thd", "round_to_clip",
            "gmiou_cls_threshold", "cls_thresholds", "eval_k_list", "max_pred_windows",
            "exist_score_mode", "gate_recall_thd", "zero_decision_thd",
            "zero_veto_thd", "zero_localization_thd",
            "set_selection_mode", "selection_k", "selection_max_output",
            "pairwise_redundancy_lambda", "count_prior_weight", "selection_stop_thd",
            "same_event_thd", "fusion_boundary_std_thd", "pairwise_detach_inputs",
        ):
            if hasattr(saved_opt, name):
                setattr(opt, name, getattr(saved_opt, name))
    for name, value in getattr(opt, "cli_overrides", {}).items():
        setattr(opt, name, value)

    # Dataset construction must happen after the checkpoint protocol is
    # restored; clean checkpoints may trim CLIP padding while release models
    # intentionally retain the original fixed-length layout.
    load_labels = opt.eval_split_name == "val"
    eval_dataset = StartEndDataset(**build_dataset_config(
        opt, opt.eval_path, load_labels=load_labels
    ))
    expected_examples = sum(1 for _ in open(opt.eval_path, "r", encoding="utf-8"))
    if len(eval_dataset) != expected_examples:
        raise RuntimeError(
            f"Incomplete feature coverage: eval={len(eval_dataset)}/{expected_examples}"
        )

    model, criterion, _, _ = setup_model(opt)
    model.load_state_dict(ckpt_state)
    logger.info("Model checkpoint: %s", opt.model_path)
    if not load_labels:
        criterion = None

    save_submission_filename = f"moment_detr_gmr_{opt.eval_split_name}_submission.jsonl"
    with torch.no_grad():
        metrics, _, _ = eval_epoch(None, model, eval_dataset, opt, save_submission_filename, criterion)

    if opt.eval_split_name == "val" and metrics is not None:
        logger.info("metrics_no_nms %s", pprint.pformat(metrics["brief"], indent=4))

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Moment-DETR-GMR inference on Soccer-GMR features.",
        allow_abbrev=False,
    )
    parser.add_argument("--model", "-m", default="moment_detr", choices=["moment_detr"])
    parser.add_argument("--dataset", "-d", default="soccer_gmr", choices=["soccer_gmr"])
    parser.add_argument("--feature", "-f", default="clip_slowfast", choices=["clip_slowfast"])
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--split", type=str, required=True, choices=["val", "test"])
    parser.add_argument("--eval_path", type=str, required=True)
    parser.add_argument("--t_feat_dir", type=str, default=None)
    parser.add_argument("--v_feat_dirs", type=str, nargs="+", default=None)
    parser.add_argument("--results_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--decode_mode", choices=["full", "threshold", "adaptive", "hard"], default=None
    )
    parser.add_argument("--diversity_lambda", type=float, default=None)
    parser.add_argument("--quality_score_alpha", type=float, default=None)
    parser.add_argument("--count_confidence_thd", type=float, default=None)
    parser.add_argument("--count_exist_thd", type=float, default=None)
    parser.add_argument("--window_score_thd", type=float, default=None)
    parser.add_argument("--gmiou_cls_threshold", type=float, default=None)
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
    parser.add_argument(
        "--trim_text_by_attention_mask", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--round_to_clip", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--save_raw_queries", action=argparse.BooleanOptionalAction, default=None)
    return parser


def parse_args(argv=None):
    return build_parser().parse_args(argv)

if __name__ == "__main__":
    args = parse_args()
    option_manager = BaseOptions(args.model, args.dataset, args.feature, resume=None)
    option_manager.parse()
    opt = option_manager.option

    opt.cli_overrides = {}
    for name in [
        "eval_path", "t_feat_dir", "results_dir", "device", "decode_mode",
        "diversity_lambda", "quality_score_alpha", "count_exist_thd", "count_confidence_thd",
        "window_score_thd", "gmiou_cls_threshold", "trim_text_by_attention_mask",
        "round_to_clip", "save_raw_queries", "exist_score_mode", "gate_recall_thd",
        "zero_decision_thd", "zero_veto_thd",
        "zero_localization_thd", "set_selection_mode", "selection_k",
        "selection_max_output", "pairwise_redundancy_lambda", "count_prior_weight",
        "selection_stop_thd", "same_event_thd", "fusion_boundary_std_thd",
    ]:
        value = getattr(args, name)
        if value is not None:
            setattr(opt, name, value)
            opt.cli_overrides[name] = value
    if args.v_feat_dirs is not None:
        opt.v_feat_dirs = args.v_feat_dirs
    if args.results_dir is not None:
        os.makedirs(opt.results_dir, exist_ok=True)
    else:
        os.makedirs(opt.results_dir, exist_ok=True)

    opt.model_path = args.model_path
    opt.eval_split_name = args.split
    start_inference(opt)
