#!/usr/bin/env bash
set -euo pipefail

variant="${1:?usage: $0 <md_gmr|md_hiea2m> <gpu_id> [resume_ckpt] [run_suffix]}"
gpu_id="${2:?usage: $0 <md_gmr|md_hiea2m> <gpu_id> [resume_ckpt] [run_suffix]}"

case "$variant" in
  md_gmr|md_hiea2m) ;;
  *) echo "unsupported strict Moment variant: $variant" >&2; exit 2 ;;
esac

base_dir="artifacts/formal_strict/moment_detr/seed2023/${variant}_b128"
resume_ckpt="${3:-${base_dir}/best_map.ckpt}"
run_suffix="${4:-rerun_from_best}"
run_dir="artifacts/formal_strict/moment_detr/seed2023/${variant}_b128_${run_suffix}"

if [[ ! -f "$resume_ckpt" ]]; then
  echo "missing interrupted-run checkpoint: $resume_ckpt" >&2
  exit 1
fi
if [[ -e "$run_dir" ]]; then
  echo "refusing to overwrite existing rerun directory: $run_dir" >&2
  exit 1
fi

# train.py's --resume is a weights-only fine-tuning load, not an optimizer/epoch
# continuation.  Keep this run separate so its provenance is unambiguous.
CUDA_VISIBLE_DEVICES="$gpu_id" python -u training/moment_detr_gmr/train.py \
  --variant "$variant" --resume "$resume_ckpt" \
  --seed 2023 --lr 5e-5 --n_epoch 200 --max_es_cnt 50 --eval_epoch_interval 1 \
  --bsz 128 --eval_bsz 128 --num_workers 4 --selection_metric mAP \
  --no-trim_text_by_attention_mask --round_to_clip --mask-null-vmr-loss \
  --results_dir "$run_dir" --device cuda
