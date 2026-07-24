#!/usr/bin/env bash
# Start Strict CG-DETR GMR training with batch size 32
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

out_dir="artifacts/strict_bsz32/cg_detr/seed2023/cg_detr_gmr"
mkdir -p "$out_dir"

exec env CUDA_VISIBLE_DEVICES=1 python -u -m methods.cg_detr_gmr.train \
  --variant cg_detr_gmr \
  --seed 2023 --epochs 200 --batch_size 32 --eval_bsz 32 \
  --patience 50 --eval_interval 1 --num_workers 0 --lr 3e-5 \
  --train_sample_mode mixed --map_num_workers 1 \
  --reference_map 4.94 --reference_gmiou3 1.56 \
  --train_annotation data/label/Standard/train.jsonl \
  --eval_annotation data/label/Standard/val.jsonl \
  --video_feature_dirs Soccer-GMR/feature/standard/clip Soccer-GMR/feature/standard/slowfast \
  --text_feature_dir Soccer-GMR/feature/standard/clip_text \
  --output_dir "$out_dir" --device cuda --round_to_clip \
  --init_checkpoint artifacts/formal/cg_detr/seed2023/cg_detr/best_map.ckpt \
  --mask-null-vmr-loss > "$out_dir/stdout.log" 2>&1
