#!/usr/bin/env bash
# Normal (non-nohup) strict-null Moment-DETR paired run at batch size 128.
set -euo pipefail
if [[ $# -ne 2 ]]; then
  echo "Usage: $0 {md_gmr|md_quality|md_hiea2m} GPU_ID" >&2
  exit 2
fi
variant="$1"
gpu_id="$2"
case "$variant" in md_gmr|md_quality|md_hiea2m) ;; *) echo "invalid variant" >&2; exit 2;; esac
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
run_dir="artifacts/formal_strict/moment_detr/seed2023/${variant}_b128"
exec env CUDA_VISIBLE_DEVICES="$gpu_id" python -u training/moment_detr_gmr/train.py \
  --variant "$variant" --resume Soccer-GMR/checkpoint/moment_detr_gmr \
  --seed 2023 --lr 5e-5 --n_epoch 200 --max_es_cnt 50 \
  --eval_epoch_interval 1 --bsz 128 --eval_bsz 128 --num_workers 4 \
  --selection_metric mAP --no-trim_text_by_attention_mask --round_to_clip \
  --mask-null-vmr-loss --results_dir "$run_dir" --device cuda
