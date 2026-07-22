#!/usr/bin/env bash
# Continue the two interrupted formal tracks that already use batch size 128.
# This script intentionally uses ordinary foreground Python processes (no nohup)
# and is meant to be run from a persistent terminal/screen session.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

qd_dir="artifacts/formal/qd_detr/seed2023/qd_detr_gmr"
eatr_dir="artifacts/formal/eatr/seed2023/eatr_gmr"

CUDA_VISIBLE_DEVICES=0 python -u -m methods.qd_detr_gmr.train \
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
  --resume "$qd_dir/latest.ckpt" > "$qd_dir/resume_b128.stdout.log" 2>&1 &
qd_pid="$!"
printf '%s\n' "$qd_pid" > "$qd_dir/resume_b128.pid"

CUDA_VISIBLE_DEVICES=1 python -u -m methods.eatr_gmr.train \
  --variant eatr_gmr \
  --train-annotations data/label/Standard/train.jsonl \
  --val-annotations data/label/Standard/val.jsonl \
  --slowfast-dir Soccer-GMR/feature/standard/slowfast \
  --clip-dir Soccer-GMR/feature/standard/clip \
  --text-dir Soccer-GMR/feature/standard/clip_text \
  --seed 2023 --epochs 200 --eval-interval 1 --patience 50 \
  --batch-size 128 --eval-batch-size 128 --lr 3e-5 --num-workers 0 \
  --train-sample-mode mixed --reference-map 7.92 --reference-gmiou3 2.99 \
  --output-dir "$eatr_dir" --device cuda --resume "$eatr_dir/last.pt" \
  > "$eatr_dir/resume_b128.stdout.log" 2>&1 &
eatr_pid="$!"
printf '%s\n' "$eatr_pid" > "$eatr_dir/resume_b128.pid"

printf 'QD-DETR-GMR PID=%s (GPU0)\n' "$qd_pid"
printf 'EaTR-GMR PID=%s (GPU1)\n' "$eatr_pid"
wait "$qd_pid" "$eatr_pid"
