#!/usr/bin/env bash
set -euo pipefail

stage="${1:?usage: $0 <stage1_geometry|stage2_zero|stage2_calibrate|stage3_pairwise|stage4_5_selection> [gpu_id]}"
gpu_id="${2:-0}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python_bin="${PYTHON_BIN:-/home/guoxiangyu/miniconda3/bin/python}"
train_path="${TRAIN_PATH:-data/label/Standard/train.jsonl}"
val_path="${VAL_PATH:-data/label/Standard/val.jsonl}"
text_dir="${TEXT_DIR:-Soccer-GMR/feature/standard/clip_text}"
clip_dir="${CLIP_DIR:-Soccer-GMR/feature/standard/clip}"
slowfast_dir="${SLOWFAST_DIR:-Soccer-GMR/feature/standard/slowfast}"
root="${SELECTOR_ROOT:-artifacts/validation_selector_ablation/seed2023}"
base_checkpoint="${BASE_CHECKPOINT:-artifacts/formal_strict/moment_detr/seed2023/md_hiea2m_b128_rerun_from_best_v2/best_joint.ckpt}"
selector_seed="${SELECTOR_SEED:-2023}"
zero_dir="$root/stage2_zero"
pairwise_dir="$root/stage3_pairwise"

mkdir -p "$root"

case "$stage" in
  stage1_geometry)
    source_prediction="${GEOMETRY_PREDICTION:-artifacts/formal_strict/moment_detr/seed2023/md_hiea2m_b128_rerun_from_best_v2/best_map_soccer_gmr_val_preds.jsonl}"
    output_dir="$root/stage1_geometry"
    mkdir -p "$output_dir"
    "$python_bin" -u scripts/ablate_temporal_dedup.py \
      --prediction-path "$source_prediction" \
      --gt-path "$val_path" \
      --output-dir "$output_dir" \
      --methods none hard_nms diou_nms soft_nms_linear soft_nms_gaussian cluster_representative cluster_fusion \
      --iou-thresholds 0.5 0.7 0.9 \
      --soft-sigmas 0.5 1.0 2.0 \
      --selection-budgets 1 3 5 predicted_count \
      --map-num-workers 1 2>&1 | tee "$output_dir/run.log"
    ;;

  stage2_zero)
    mkdir -p "$zero_dir"
    CUDA_VISIBLE_DEVICES="$gpu_id" "$python_bin" -u training/moment_detr_gmr/train.py \
      --variant md_hiea2m_zero \
      --resume "$base_checkpoint" \
      --seed "$selector_seed" \
      --trainable_scope zero \
      --exist_score_mode zero \
      --set_selection_mode legacy \
      --selection_metric gmiou3 \
      --n_epoch "${ZERO_EPOCHS:-80}" \
      --max_es_cnt "${ZERO_PATIENCE:-15}" \
      --bsz 128 --eval_bsz 128 --num_workers 0 \
      --lr "${ZERO_LR:-0.0001}" \
      --zero_positive_query_weight "${ZERO_POSITIVE_WEIGHT:-2.0}" \
      --mask-null-vmr-loss \
      --no-trim_text_by_attention_mask --round_to_clip \
      --train_path "$train_path" --eval_path "$val_path" \
      --t_feat_dir "$text_dir" --v_feat_dirs "$clip_dir" "$slowfast_dir" \
      --results_dir "$zero_dir" --device cuda 2>&1 | tee "$zero_dir/stdout.log"
    ;;

  stage2_calibrate)
    prediction="${ZERO_PREDICTION:-$zero_dir/best_soccer_gmr_val_preds.jsonl}"
    if [[ ! -f "$prediction" ]]; then
      echo "missing stage-2 validation prediction: $prediction" >&2
      exit 2
    fi
    "$python_bin" -u scripts/calibrate_two_stage_gate.py \
      --prediction-path "$prediction" --gt-path "$val_path" \
      --output "$root/stage2_gate_calibration.json" \
      --minimum-positive-pass-rate "${MIN_POSITIVE_PASS_RATE:-0.95}" \
      --map-num-workers 1 2>&1 | tee "$root/stage2_gate_calibration.log"
    ;;

  stage3_pairwise)
    zero_checkpoint="${ZERO_CHECKPOINT:-$zero_dir/best.ckpt}"
    if [[ ! -f "$zero_checkpoint" ]]; then
      echo "missing stage-2 checkpoint: $zero_checkpoint" >&2
      exit 2
    fi
    mkdir -p "$pairwise_dir"
    CUDA_VISIBLE_DEVICES="$gpu_id" "$python_bin" -u training/moment_detr_gmr/train.py \
      --variant md_hiea2m_pairwise \
      --resume "$zero_checkpoint" \
      --seed "${PAIRWISE_SEED:-$selector_seed}" \
      --trainable_scope pairwise \
      --exist_score_mode cascade \
      --set_selection_mode learned_topk --selection_k 3 \
      --pairwise_redundancy_lambda "${PAIRWISE_LAMBDA:-1.0}" \
      --selection_metric mAP \
      --n_epoch "${PAIRWISE_EPOCHS:-80}" \
      --max_es_cnt "${PAIRWISE_PATIENCE:-15}" \
      --bsz 128 --eval_bsz 128 --num_workers 0 \
      --lr "${PAIRWISE_LR:-0.0001}" \
      --pair_assignment_iou "${PAIR_ASSIGNMENT_IOU:-0.3}" \
      --pair_ambiguity_margin "${PAIR_AMBIGUITY_MARGIN:-0.05}" \
      --pair_hard_negative_weight "${PAIR_HARD_NEGATIVE_WEIGHT:-2.0}" \
      --mask-null-vmr-loss \
      --no-trim_text_by_attention_mask --round_to_clip \
      --train_path "$train_path" --eval_path "$val_path" \
      --t_feat_dir "$text_dir" --v_feat_dirs "$clip_dir" "$slowfast_dir" \
      --results_dir "$pairwise_dir" --device cuda 2>&1 | tee "$pairwise_dir/stdout.log"
    ;;

  stage4_5_selection)
    pairwise_checkpoint="${PAIRWISE_CHECKPOINT:-$pairwise_dir/best.ckpt}"
    if [[ ! -f "$pairwise_checkpoint" ]]; then
      echo "missing stage-3 checkpoint: $pairwise_checkpoint" >&2
      exit 2
    fi
    raw_dir="$pairwise_dir/raw_val"
    output_dir="$root/stage4_5_selection"
    mkdir -p "$raw_dir" "$output_dir"
    gate_recall="${GATE_RECALL_THD:-0.3}"
    zero_decision="${ZERO_DECISION_THD:-0.6}"
    zero_veto="${ZERO_VETO_THD:-0.7}"
    zero_localization="${ZERO_LOCALIZATION_THD:-0.2}"
    gate_manifest="$root/stage2_gate_calibration.json"
    if [[ -f "$gate_manifest" ]]; then
      read -r gate_recall zero_decision zero_veto zero_localization < <(
        "$python_bin" -c "import json; c=json.load(open('$gate_manifest'))['selected']['config']; print(c['gate_recall_thd'], c['zero_decision_thd'], c['zero_veto_thd'], c['zero_localization_thd'])"
      )
    fi
    CUDA_VISIBLE_DEVICES="$gpu_id" "$python_bin" -u training/moment_detr_gmr/evaluate.py \
      --model_path "$pairwise_checkpoint" \
      --split val --eval_path "$val_path" \
      --t_feat_dir "$text_dir" --v_feat_dirs "$clip_dir" "$slowfast_dir" \
      --results_dir "$raw_dir" --device cuda \
      --exist_score_mode cascade --set_selection_mode legacy \
      --gate_recall_thd "$gate_recall" --zero_decision_thd "$zero_decision" \
      --zero_veto_thd "$zero_veto" --zero_localization_thd "$zero_localization" \
      --save_raw_queries 2>&1 | tee "$raw_dir/stdout.log"
    "$python_bin" -u scripts/ablate_learned_selector.py \
      --prediction-path "$raw_dir/moment_detr_gmr_val_submission.jsonl" \
      --gt-path "$val_path" --output-dir "$output_dir" \
      --fixed-k 3 --max-output 10 --map-num-workers 1 2>&1 | tee "$output_dir/run.log"
    ;;

  *)
    echo "unknown stage: $stage" >&2
    exit 2
    ;;
esac
