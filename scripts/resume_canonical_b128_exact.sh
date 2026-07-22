#!/usr/bin/env bash
set -euo pipefail

task="${1:?usage: $0 <qd|cg> <gpu_id>}"
gpu_id="${2:?missing gpu id}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

case "$task" in
  qd)
    run_dir="artifacts/canonical_b128_restart/qd_detr/seed2023/qd_detr"
    exec env CUDA_VISIBLE_DEVICES="$gpu_id" python -u -m methods.qd_detr_gmr.train \
      --variant qd_detr --seed 2023 --epochs 200 --batch_size 128 --eval_bsz 128 \
      --patience 200 --eval_interval 1 --num_workers 0 --lr 3e-5 \
      --train_sample_mode positive --map_num_workers 1 \
      --train_annotation data/label/Standard/train.jsonl \
      --eval_annotation data/label/Standard/val.jsonl \
      --video_feature_dirs Soccer-GMR/feature/standard/slowfast Soccer-GMR/feature/standard/clip \
      --text_feature_dir Soccer-GMR/feature/standard/clip_text \
      --output_dir "$run_dir" --device cuda --round_to_clip --no-diagnostic_decoders \
      --resume "$run_dir/latest.ckpt"
    ;;
  cg)
    run_dir="artifacts/canonical_b128_restart/cg_detr/seed2023/cg_detr"
    exec env CUDA_VISIBLE_DEVICES="$gpu_id" python -u -m methods.cg_detr_gmr.train \
      --variant cg_detr --seed 2023 --epochs 200 --batch_size 128 --eval_bsz 128 \
      --patience 200 --eval_interval 1 --num_workers 0 --lr 3e-5 \
      --train_sample_mode positive --map_num_workers 1 \
      --train_annotation data/label/Standard/train.jsonl \
      --eval_annotation data/label/Standard/val.jsonl \
      --video_feature_dirs Soccer-GMR/feature/standard/slowfast Soccer-GMR/feature/standard/clip \
      --text_feature_dir Soccer-GMR/feature/standard/clip_text \
      --output_dir "$run_dir" --device cuda --round_to_clip \
      --resume "$run_dir/latest.ckpt"
    ;;
  *)
    echo "unknown canonical task: $task" >&2
    exit 2
    ;;
esac
