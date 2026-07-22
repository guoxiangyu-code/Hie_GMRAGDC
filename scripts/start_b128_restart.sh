#!/usr/bin/env bash
# Start a new bsz=128 continuation from the latest weights of a run whose
# original batch size was 32 or 64.  This is deliberately a weights-only
# restart, not a claimed exact optimizer-state resume, and never uses nohup.
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 {canonical_qd|formal_cg|canonical_cg|canonical_eatr} GPU_ID" >&2
  exit 2
fi
task="$1"
gpu_id="$2"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

case "$task" in
  canonical_qd)
    exec env CUDA_VISIBLE_DEVICES="$gpu_id" python -u -m methods.qd_detr_gmr.train \
      --variant qd_detr --seed 2023 --epochs 200 --batch_size 128 --eval_bsz 128 \
      --patience 200 --eval_interval 1 --num_workers 0 --lr 3e-5 \
      --train_sample_mode positive --map_num_workers 1 \
      --train_annotation data/label/Standard/train.jsonl \
      --eval_annotation data/label/Standard/val.jsonl \
      --video_feature_dirs Soccer-GMR/feature/standard/slowfast Soccer-GMR/feature/standard/clip \
      --text_feature_dir Soccer-GMR/feature/standard/clip_text \
      --output_dir artifacts/canonical_b128_restart/qd_detr/seed2023/qd_detr \
      --device cuda --round_to_clip --no-diagnostic_decoders \
      --init_checkpoint artifacts/canonical/qd_detr/seed2023/qd_detr/latest.ckpt
    ;;
  formal_cg)
    exec env CUDA_VISIBLE_DEVICES="$gpu_id" python -u -m methods.cg_detr_gmr.train \
      --variant cg_detr --seed 2023 --epochs 200 --batch_size 128 --eval_bsz 128 \
      --patience 50 --eval_interval 1 --num_workers 0 --lr 1e-4 \
      --train_sample_mode positive --map_num_workers 1 \
      --train_annotation data/label/Standard/train.jsonl \
      --eval_annotation data/label/Standard/val.jsonl \
      --video_feature_dirs Soccer-GMR/feature/standard/clip Soccer-GMR/feature/standard/slowfast \
      --text_feature_dir Soccer-GMR/feature/standard/clip_text \
      --output_dir artifacts/formal_b128_restart/cg_detr/seed2023/cg_detr \
      --device cuda --round_to_clip \
      --init_checkpoint artifacts/formal/cg_detr/seed2023/cg_detr/latest.ckpt
    ;;
  canonical_cg)
    exec env CUDA_VISIBLE_DEVICES="$gpu_id" python -u -m methods.cg_detr_gmr.train \
      --variant cg_detr --seed 2023 --epochs 200 --batch_size 128 --eval_bsz 128 \
      --patience 200 --eval_interval 1 --num_workers 0 --lr 3e-5 \
      --train_sample_mode positive --map_num_workers 1 \
      --train_annotation data/label/Standard/train.jsonl \
      --eval_annotation data/label/Standard/val.jsonl \
      --video_feature_dirs Soccer-GMR/feature/standard/slowfast Soccer-GMR/feature/standard/clip \
      --text_feature_dir Soccer-GMR/feature/standard/clip_text \
      --output_dir artifacts/canonical_b128_restart/cg_detr/seed2023/cg_detr \
      --device cuda --round_to_clip \
      --init_checkpoint artifacts/canonical/cg_detr/seed2023/cg_detr/latest.ckpt
    ;;
  canonical_eatr)
    exec env CUDA_VISIBLE_DEVICES="$gpu_id" python -u -m methods.eatr_gmr.train \
      --variant eatr \
      --train-annotations data/label/Standard/train.jsonl \
      --val-annotations data/label/Standard/val.jsonl \
      --slowfast-dir Soccer-GMR/feature/standard/slowfast \
      --clip-dir Soccer-GMR/feature/standard/clip \
      --text-dir Soccer-GMR/feature/standard/clip_text \
      --seed 2023 --epochs 200 --eval-interval 1 --patience 200 \
      --batch-size 128 --eval-batch-size 128 --lr 3e-5 --num-workers 0 \
      --train-sample-mode positive \
      --output-dir artifacts/canonical_b128_restart/eatr/seed2023/eatr \
      --device cuda --init-checkpoint artifacts/canonical/eatr/seed2023/eatr/last.pt
    ;;
  *)
    echo "Unknown task: $task" >&2
    exit 2
    ;;
esac
