#!/usr/bin/env bash
# Normal (non-nohup) exact continuation for the interrupted bsz=128 QD-GMR run.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
qd_dir="artifacts/formal/qd_detr/seed2023/qd_detr_gmr"

exec env CUDA_VISIBLE_DEVICES=0 python -u -m methods.qd_detr_gmr.train \
  --variant qd_detr_gmr \
  --seed 2023 --epochs 200 --batch_size 128 --eval_bsz 128 \
  --patience 50 --eval_interval 1 --num_workers 0 --lr 3e-5 \
  --train_sample_mode mixed --map_num_workers 1 \
  --reference_map 7.32 --reference_gmiou3 2.6 \
  --train_annotation data/label/Standard/train.jsonl \
  --eval_annotation data/label/Standard/val.jsonl \
  --video_feature_dirs Soccer-GMR/feature/standard/clip Soccer-GMR/feature/standard/slowfast \
  --text_feature_dir Soccer-GMR/feature/standard/clip_text \
  --output_dir "$qd_dir" --device cuda --round_to_clip --no-diagnostic_decoders \
  --resume "$qd_dir/latest.ckpt" > "$qd_dir/resume_b128.stdout.log" 2>&1
