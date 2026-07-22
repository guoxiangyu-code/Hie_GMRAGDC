"""Shared command-line options for Soccer-GMR CG-DETR runs."""

from __future__ import annotations

import argparse
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FEATURE_ROOT = REPO_ROOT / "Soccer-GMR" / "feature" / "standard"

VARIANT_FLAGS = {
    "cg_detr": (False, False, False, False),
    "cg_detr_gmr": (True, False, False, False),
    "cg_quality": (True, True, False, False),
    "cg_phrase": (True, False, True, False),
    "cg_counter": (True, False, False, True),
    "cg_hiea2m": (True, True, True, True),
}


def detect_variant(state_dict: dict) -> tuple[str, dict[str, bool]]:
    structure = {
        "exist": any(name.startswith("exist_head.") for name in state_dict),
        "quality": any(name.startswith("quality_embed.") for name in state_dict),
        "phrase": any(name.startswith("phrase_grounding.") for name in state_dict),
        "counter": any(name.startswith("hierarchical_counter.") for name in state_dict),
    }
    flags = tuple(structure[name] for name in ("exist", "quality", "phrase", "counter"))
    matches = [name for name, expected in VARIANT_FLAGS.items() if expected == flags]
    if len(matches) != 1:
        raise ValueError(f"unsupported CG checkpoint structure: {structure}")
    return matches[0], structure


def add_data_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--train_annotation", default=str(REPO_ROOT / "data/label/Standard/train.jsonl"))
    parser.add_argument("--eval_annotation", default=str(REPO_ROOT / "data/label/Standard/val.jsonl"))
    parser.add_argument(
        "--video_feature_dirs", nargs="+",
        default=[str(DEFAULT_FEATURE_ROOT / "clip"), str(DEFAULT_FEATURE_ROOT / "slowfast")],
    )
    parser.add_argument("--text_feature_dir", default=str(DEFAULT_FEATURE_ROOT / "clip_text"))
    parser.add_argument("--max_q_l", type=int, default=32)
    parser.add_argument("--max_v_l", type=int, default=75)
    parser.add_argument("--max_windows", type=int, default=10)
    parser.add_argument("--clip_length", type=float, default=2.0)
    parser.add_argument("--trim_text_by_attention_mask", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use_tef", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--eval_bsz", type=int, default=32)


def add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--variant", choices=tuple(VARIANT_FLAGS),
        default="cg_detr_gmr",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--v_feat_dim", type=int, default=2818)
    parser.add_argument("--t_feat_dim", type=int, default=512)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dim_feedforward", type=int, default=1024)
    parser.add_argument("--enc_layers", type=int, default=3)
    parser.add_argument("--dec_layers", type=int, default=3)
    parser.add_argument("--t2v_layers", type=int, default=2)
    parser.add_argument("--sent_layers", type=int, default=1)
    parser.add_argument("--moment_layers", type=int, default=1)
    parser.add_argument("--dummy_layers", type=int, default=2)
    parser.add_argument("--nheads", type=int, default=8)
    parser.add_argument("--num_queries", type=int, default=10)
    parser.add_argument("--num_dummies", type=int, default=45)
    parser.add_argument("--total_prompts", type=int, default=10)
    parser.add_argument("--num_prompts", type=int, default=1)
    parser.add_argument("--input_dropout", type=float, default=0.5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--n_input_proj", type=int, default=2)
    parser.add_argument("--position_embedding", default="sine", choices=("sine",))
    parser.add_argument("--pre_norm", action="store_true")
    parser.add_argument("--use_txt_pos", action="store_true")
    parser.add_argument("--aux_loss", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--mask-null-vmr-loss", "--mask_null_vmr_loss",
        dest="mask_null_vmr_loss",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "apply the GMR indicator I(y=1) to localization/query-quality "
            "losses so null queries supervise only GMR decision heads"
        ),
    )
    parser.add_argument("--use_cg_aux", action=argparse.BooleanOptionalAction, default=True,
                        help="retain moment/sentence, distillation, and orthogonality losses derived from MR windows")
    parser.add_argument("--exist_hidden_dim", type=int, default=None)
    parser.add_argument("--exist_loss_coef", type=float, default=1.0)
    parser.add_argument("--span_loss_type", default="l1", choices=("l1",))

    parser.add_argument("--set_cost_span", type=float, default=10.0)
    parser.add_argument("--set_cost_giou", type=float, default=1.0)
    parser.add_argument("--set_cost_class", type=float, default=4.0)
    parser.add_argument("--span_loss_coef", type=float, default=10.0)
    parser.add_argument("--giou_loss_coef", type=float, default=1.0)
    parser.add_argument("--label_loss_coef", type=float, default=4.0)
    parser.add_argument("--eos_coef", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--saliency_margin", type=float, default=0.2)
    parser.add_argument("--lw_ms_align", type=float, default=1.0)
    parser.add_argument("--lw_distill", type=float, default=1.0)

    # Unified HieA2M heads.  CG already owns sentence/dummy ACA; the local
    # DualGround migration is phrase-only by construction.
    parser.add_argument("--quality_loss_coef", type=float, default=1.0)
    parser.add_argument("--quality_alpha", type=float, default=0.5)
    parser.add_argument("--phrase_num_phrases", type=int, default=3)
    parser.add_argument("--phrase_slot_iterations", type=int, default=1)
    parser.add_argument("--phrase_gate_init", type=float, default=-4.0)
    parser.add_argument("--phrase_dqa_scale", type=float, default=0.3)
    parser.add_argument("--phrase_dqa_loss_coef", type=float, default=0.05)
    parser.add_argument("--phrase_eos_loss_coef", type=float, default=0.1)
    parser.add_argument("--phrase_eos_temperature", type=float, default=0.07)
    parser.add_argument("--counter_dropout", type=float, default=0.1)
    parser.add_argument("--counter_detach_scores", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--count_loss_coef", type=float, default=1.0)
    parser.add_argument("--count_ordinal_loss_coef", type=float, default=0.25)
    parser.add_argument("--count_contrastive_loss_coef", type=float, default=0.05)
    parser.add_argument("--count_consistency_loss_coef", type=float, default=0.05)
    parser.add_argument("--counter_contrastive_temperature", type=float, default=0.1)
    parser.add_argument(
        "--positive_count_class_counts", type=int, nargs=4,
        default=[1423, 565, 117, 31], metavar=("N1", "N2", "N3", "N4PLUS"),
    )
    parser.add_argument(
        "--decode_mode", choices=("full", "threshold", "adaptive", "hard"),
        default="full",
    )
    parser.add_argument("--existence_threshold", type=float, default=0.4)
    parser.add_argument("--count_confidence_threshold", type=float, default=0.55)
    parser.add_argument("--window_score_threshold", type=float, default=0.1)
    parser.add_argument("--diversity_lambda", type=float, default=0.0)


def finalize_model_arguments(args: argparse.Namespace) -> argparse.Namespace:
    (
        args.use_exist_head,
        args.use_quality_head,
        args.use_phrase_grounding,
        args.use_hierarchical_counter,
    ) = VARIANT_FLAGS[args.variant]
    # Soccer-GMR has moment windows but no QVHighlights saliency ratings.  The
    # upstream saliency ranking/negative-pair branch is therefore unavailable.
    args.use_saliency = False
    args.lw_saliency = 0.0
    args.lw_wattn = 0.0
    args.contrastive_align_loss = False
    args.contrastive_hdim = 64
    args.contrastive_align_loss_coef = 0.0
    args.a_feat_dir = None
    args.dset_name = "soccer_gmr"
    expected_dim = 2816 + (2 if args.use_tef else 0)
    if args.v_feat_dim != expected_dim:
        raise ValueError(
            f"v_feat_dim={args.v_feat_dim}, but CLIP+SlowFast with use_tef={args.use_tef} "
            f"requires {expected_dim}"
        )
    if args.hidden_dim % args.nheads:
        raise ValueError("hidden_dim must be divisible by nheads")
    if not 1 <= args.num_prompts <= args.total_prompts:
        raise ValueError("num_prompts must lie in [1, total_prompts]")
    return args
