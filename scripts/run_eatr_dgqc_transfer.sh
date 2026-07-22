#!/usr/bin/env bash
set -euo pipefail

stage="${1:?usage: $0 <gmr|resume_gmr|quality|dual|counter|hiea2m|children|full|resume_full> [gpu_id]}"
gpu_id="${2:-1}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python_bin="${PYTHON_BIN:-/home/guoxiangyu/miniconda3/bin/python}"
root="${EATR_DGQC_ROOT:-artifacts/eatr_dgqc_transfer/seed2023}"
plain_parent="${EATR_PLAIN_PARENT:-$root/frozen_parent/eatr_plain_b128_best_map.pt}"
gmr_dir="$root/eatr_gmr_strict"
seed="${EATR_SEED:-2023}"

train_annotations="${TRAIN_ANNOTATIONS:-data/label/Standard/train.jsonl}"
val_annotations="${VAL_ANNOTATIONS:-data/label/Standard/val.jsonl}"
slowfast_dir="${SLOWFAST_DIR:-Soccer-GMR/feature/standard/slowfast}"
clip_dir="${CLIP_DIR:-Soccer-GMR/feature/standard/clip}"
text_dir="${TEXT_DIR:-Soccer-GMR/feature/standard/clip_text}"

plain_reference_map="${PLAIN_REFERENCE_MAP:-8.73}"
plain_reference_gmiou3="${PLAIN_REFERENCE_GMIOU3:-3.51}"

common_args=(
  --train-annotations "$train_annotations"
  --val-annotations "$val_annotations"
  --slowfast-dir "$slowfast_dir"
  --clip-dir "$clip_dir"
  --text-dir "$text_dir"
  --seed "$seed"
  --epochs "${EATR_EPOCHS:-200}"
  --eval-interval 1
  --patience "${EATR_PATIENCE:-50}"
  --batch-size 128
  --eval-batch-size 128
  --lr "${EATR_LR:-3e-5}"
  --backbone-lr-scale "${EATR_BACKBONE_LR_SCALE:-0.1}"
  --num-workers 0
  --train-sample-mode mixed
  --mask-null-vmr-loss
  --device cuda
)

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "missing required checkpoint: $1" >&2
    exit 2
  fi
}

read_gmr_references() {
  local metrics_path="$gmr_dir/best_joint_val_metrics.json"
  if [[ ! -f "$metrics_path" ]]; then
    echo "missing matched GMR metrics: $metrics_path" >&2
    exit 2
  fi
  "$python_bin" -c \
    "import json; b=json.load(open('$metrics_path'))['brief']; print(b['mAP'], b['G-mIoU@3'])"
}

run_gmr() {
  require_file "$plain_parent"
  mkdir -p "$gmr_dir"
  CUDA_VISIBLE_DEVICES="$gpu_id" "$python_bin" -u -m methods.eatr_gmr.train \
    --variant eatr_gmr \
    --init-checkpoint "$plain_parent" \
    --reference-map "$plain_reference_map" \
    --reference-gmiou3 "$plain_reference_gmiou3" \
    --output-dir "$gmr_dir" \
    "${common_args[@]}" 2>&1 | tee "$gmr_dir/stdout.log"
}

resume_gmr() {
  local checkpoint="$gmr_dir/last.pt"
  require_file "$checkpoint"
  CUDA_VISIBLE_DEVICES="$gpu_id" "$python_bin" -u -m methods.eatr_gmr.train \
    --variant eatr_gmr \
    --resume "$checkpoint" \
    --reference-map "$plain_reference_map" \
    --reference-gmiou3 "$plain_reference_gmiou3" \
    --output-dir "$gmr_dir" \
    "${common_args[@]}" 2>&1 | tee -a "$gmr_dir/stdout.log"
}

run_child() {
  local variant="$1"
  local parent="$gmr_dir/best.pt"
  local output_dir="$root/$variant"
  require_file "$parent"
  read -r gmr_reference_map gmr_reference_gmiou3 < <(read_gmr_references)
  mkdir -p "$output_dir"
  CUDA_VISIBLE_DEVICES="$gpu_id" "$python_bin" -u -m methods.eatr_gmr.train \
    --variant "$variant" \
    --init-checkpoint "$parent" \
    --reference-map "$gmr_reference_map" \
    --reference-gmiou3 "$gmr_reference_gmiou3" \
    --output-dir "$output_dir" \
    "${common_args[@]}" 2>&1 | tee "$output_dir/stdout.log"
}

run_children_parallel() {
  local variants=(eatr_quality eatr_dual eatr_counter eatr_hiea2m)
  local pids=()
  local variant
  for variant in "${variants[@]}"; do
    run_child "$variant" > "$root/${variant}.launcher.log" 2>&1 &
    pids+=("$!")
    echo "$!" > "$root/${variant}.pid"
    echo "started $variant pid=$! gpu=$gpu_id"
  done
  local status=0
  local pid
  for pid in "${pids[@]}"; do
    wait "$pid" || status=1
  done
  return "$status"
}

case "$stage" in
  gmr) run_gmr ;;
  resume_gmr) resume_gmr ;;
  quality) run_child eatr_quality ;;
  dual) run_child eatr_dual ;;
  counter) run_child eatr_counter ;;
  hiea2m) run_child eatr_hiea2m ;;
  children) run_children_parallel ;;
  full)
    run_gmr
    run_children_parallel
    ;;
  resume_full)
    resume_gmr
    run_children_parallel
    ;;
  *)
    echo "unknown stage: $stage" >&2
    exit 2
    ;;
esac
