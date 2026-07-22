#!/usr/bin/env bash
# Normal (non-nohup) exact continuation for a formal QD component run at bsz=128.
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 {qd_quality|qd_dual|qd_counter|qd_hiea2m} GPU_ID" >&2
  exit 2
fi
variant="$1"
gpu_id="$2"
case "$variant" in
  qd_quality|qd_dual|qd_counter|qd_hiea2m) ;;
  *) echo "Unsupported QD formal variant: $variant" >&2; exit 2 ;;
esac
if [[ ! "$gpu_id" =~ ^[0-9]+$ ]]; then
  echo "GPU_ID must be an integer" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
run_dir="artifacts/formal/qd_detr/seed2023/$variant"

exec env CUDA_VISIBLE_DEVICES="$gpu_id" python -u -m methods.qd_detr_gmr.train \
  --variant "$variant" \
  --seed 2023 --epochs 200 --batch_size 128 --eval_bsz 128 \
  --patience 50 --eval_interval 1 --num_workers 0 --lr 3e-5 \
  --train_sample_mode mixed --map_num_workers 1 \
  --reference_map 7.32 --reference_gmiou3 2.6 \
  --train_annotation data/label/Standard/train.jsonl \
  --eval_annotation data/label/Standard/val.jsonl \
  --video_feature_dirs Soccer-GMR/feature/standard/clip Soccer-GMR/feature/standard/slowfast \
  --text_feature_dir Soccer-GMR/feature/standard/clip_text \
  --output_dir "$run_dir" --device cuda --round_to_clip --no-diagnostic_decoders \
  --resume "$run_dir/latest.ckpt" > "$run_dir/resume_b128.stdout.log" 2>&1
