#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 <moment|eatr|qd|cg> <physical-gpu-id> [output-dir]" >&2
  exit 2
fi

backbone="$1"
gpu_id="$2"
if [[ ! "$gpu_id" =~ ^[0-9]+$ ]]; then
  echo "physical-gpu-id must be a non-negative integer: $gpu_id" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
python_bin="/home/guoxiangyu/miniconda3/bin/python"
seed="${STAGE_B_SEED:-2023}"
output_root="$repo_root/artifacts/cross_backbone_stage_b/seed${seed}"
cpu_threads="${GMR_CPU_THREADS:-6}"
skip_cg_marker="${STAGE_B_SKIP_CG_MARKER:-$repo_root/artifacts/supplementary_queue/seed2023/skip_cg_stage_b}"

if [[ ! "$cpu_threads" =~ ^[4-8]$ ]]; then
  echo "GMR_CPU_THREADS must be between 4 and 8: $cpu_threads" >&2
  exit 2
fi

if [[ "$backbone" == "cg" && -f "$skip_cg_marker" ]]; then
  echo "skipping CG Stage-B experiment because marker exists: $skip_cg_marker"
  exit 0
fi

case "$backbone" in
  moment)
    variant="md_quality_dual"
    default_output="$output_root/moment/$variant"
    parent="$repo_root/artifacts/formal_strict/moment_detr/seed2023/md_gmr_b128_rerun_from_best_v2/best_joint.ckpt"
    command=(
      "$python_bin" -u training/moment_detr_gmr/train.py
      --variant "$variant"
      --resume "$parent"
      --seed "$seed"
      --lr 5e-5
      --n_epoch 200
      --max_es_cnt 10
      --eval_epoch_interval 5
      --bsz 128
      --eval_bsz 128
      --num_workers 4
      --selection_metric joint
      --reference_map 7.77
      --reference_gmiou3 24.47
      --no-trim_text_by_attention_mask
      --round_to_clip
      --mask-null-vmr-loss
      --device cuda
    )
    ;;
  eatr)
    variant="eatr_quality_dual"
    default_output="$output_root/eatr/$variant"
    parent="$repo_root/artifacts/eatr_dgqc_transfer/seed2023/eatr_gmr_strict/best.pt"
    command=(
      "$python_bin" -u -m methods.eatr_gmr.train
      --variant "$variant"
      --init-checkpoint "$parent"
      --reference-map 8.02
      --reference-gmiou3 16.82
      --train-annotations "$repo_root/data/label/Standard/train.jsonl"
      --val-annotations "$repo_root/data/label/Standard/val.jsonl"
      --slowfast-dir "$repo_root/Soccer-GMR/feature/standard/slowfast"
      --clip-dir "$repo_root/Soccer-GMR/feature/standard/clip"
      --text-dir "$repo_root/Soccer-GMR/feature/standard/clip_text"
      --seed "$seed"
      --epochs 200
      --eval-interval 5
      --patience 10
      --batch-size 128
      --eval-batch-size 128
      --lr 3e-5
      --backbone-lr-scale 0.1
      --num-workers 0
      --train-sample-mode mixed
      --mask-null-vmr-loss
      --device cuda
    )
    ;;
  qd)
    variant="qd_quality_dual"
    default_output="$output_root/qd/$variant"
    parent="$repo_root/artifacts/strict_bsz32/qd_detr/seed2023/qd_detr_gmr/best_joint.ckpt"
    reference_json="$repo_root/artifacts/strict_bsz32/qd_detr/seed2023/qd_detr_gmr/best_joint_val_metrics.json"
    module_name="methods.qd_detr_gmr.train"
    ;;
  cg)
    variant="cg_quality_phrase"
    default_output="$output_root/cg/$variant"
    parent="$repo_root/artifacts/strict_bsz32/cg_detr/seed2023/cg_detr_gmr/best_joint.ckpt"
    reference_json="$repo_root/artifacts/strict_bsz32/cg_detr/seed2023/cg_detr_gmr/best_joint_val_metrics.json"
    module_name="methods.cg_detr_gmr.train"
    ;;
  *)
    echo "unknown backbone: $backbone" >&2
    exit 2
    ;;
esac

output_dir="${3:-$default_output}"
if [[ "$output_dir" != /* ]]; then
  output_dir="$repo_root/$output_dir"
fi
output_dir="$(realpath -m "$output_dir")"
case "$output_dir" in
  "$repo_root"/*) ;;
  *)
    echo "output-dir must be inside $repo_root: $output_dir" >&2
    exit 2
    ;;
esac

if [[ ! -f "$parent" ]]; then
  echo "missing strict parent checkpoint: $parent" >&2
  exit 3
fi

if [[ "$backbone" == "qd" || "$backbone" == "cg" ]]; then
  if [[ ! -f "$reference_json" ]]; then
    echo "missing strict parent metrics: $reference_json" >&2
    exit 3
  fi
  reference_map="$("$python_bin" -c "import json; print(json.load(open('$reference_json'))['brief']['mAP'])")"
  reference_gmiou3="$("$python_bin" -c "import json; print(json.load(open('$reference_json'))['brief']['G-mIoU@3'])")"
  command=(
    "$python_bin" -u -m "$module_name"
    --variant "$variant"
    --seed "$seed"
    --epochs 200
    --batch_size 32
    --eval_bsz 32
    --patience 10
    --eval_interval 5
    --num_workers 0
    --lr 3e-5
    --backbone_lr_scale 0.1
    --train_sample_mode mixed
    --map_num_workers 1
    --reference_map "$reference_map"
    --reference_gmiou3 "$reference_gmiou3"
    --train_annotation "$repo_root/data/label/Standard/train.jsonl"
    --eval_annotation "$repo_root/data/label/Standard/val.jsonl"
    --video_feature_dirs
      "$repo_root/Soccer-GMR/feature/standard/clip"
      "$repo_root/Soccer-GMR/feature/standard/slowfast"
    --text_feature_dir "$repo_root/Soccer-GMR/feature/standard/clip_text"
    --device cuda
    --round_to_clip
    --init_checkpoint "$parent"
    --mask-null-vmr-loss
  )
fi

if [[ -e "$output_dir/train_log.jsonl" || -e "$output_dir/model_latest.ckpt" ]]; then
  echo "refusing to overwrite an existing Stage-B run: $output_dir" >&2
  exit 4
fi
mkdir -p "$output_dir"
case "$backbone" in
  moment) command+=(--results_dir "$output_dir") ;;
  eatr) command+=(--output-dir "$output_dir") ;;
  qd|cg) command+=(--output_dir "$output_dir") ;;
esac

{
  printf 'backbone=%s\nvariant=%s\nseed=%s\nphysical_gpu=%s\n' \
    "$backbone" "$variant" "$seed" "$gpu_id"
  printf 'parent=%s\nstarted_at=%s\n' "$parent" "$(date --iso-8601=seconds)"
  printf 'git_commit=%s\n' "$(git rev-parse HEAD 2>/dev/null || printf unknown)"
} > "$output_dir/launch_metadata.txt"
{
  printf 'CUDA_VISIBLE_DEVICES=%q ' "$gpu_id"
  printf '%q ' "${command[@]}"
  printf '\n'
} > "$output_dir/launch_command.txt"

printf '%s\n' "$$" > "$output_dir/runner.pid"
printf 'running\n' > "$output_dir/runner.status"
finish() {
  exit_code="$?"
  if [[ "$exit_code" -eq 0 ]]; then
    printf 'completed\n' > "$output_dir/runner.status"
  else
    printf 'failed exit_code=%s\n' "$exit_code" > "$output_dir/runner.status"
  fi
}
trap finish EXIT

env CUDA_VISIBLE_DEVICES="$gpu_id" PYTHONUNBUFFERED=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  OMP_NUM_THREADS="$cpu_threads" \
  MKL_NUM_THREADS="$cpu_threads" \
  OPENBLAS_NUM_THREADS="$cpu_threads" \
  NUMEXPR_NUM_THREADS="$cpu_threads" \
  "${command[@]}" 2>&1 | tee "$output_dir/stdout.log"
