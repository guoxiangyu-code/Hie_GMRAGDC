#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

queue_root="$repo_root/artifacts/supplementary_queue/seed2023"
resume_root="$queue_root/interval5_resume"
cpu_threads="${GMR_CPU_THREADS:-6}"
mkdir -p "$resume_root"
printf 'resuming_current_wave_interval5\n' > "$queue_root/queue.status"

common_qd=(
  --seed 2023
  --epochs 200
  --batch_size 32
  --eval_bsz 32
  --patience 50
  --eval_interval 5
  --num_workers 0
  --lr 3e-5
  --backbone_lr_scale 0.1
  --train_sample_mode mixed
  --map_num_workers 1
  --reference_map 7.03
  --reference_gmiou3 3.14
  --train_annotation data/label/Standard/train.jsonl
  --eval_annotation data/label/Standard/val.jsonl
  --video_feature_dirs
    Soccer-GMR/feature/standard/clip
    Soccer-GMR/feature/standard/slowfast
  --text_feature_dir Soccer-GMR/feature/standard/clip_text
  --device cuda
  --round_to_clip
  --mask-null-vmr-loss
)

env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  OMP_NUM_THREADS="$cpu_threads" MKL_NUM_THREADS="$cpu_threads" \
  OPENBLAS_NUM_THREADS="$cpu_threads" NUMEXPR_NUM_THREADS="$cpu_threads" \
  /home/guoxiangyu/miniconda3/bin/python -u -m methods.qd_detr_gmr.train \
  --variant qd_quality \
  --output_dir artifacts/strict_bsz32/qd_detr/seed2023/qd_quality \
  --resume "$repo_root/artifacts/strict_bsz32/qd_detr/seed2023/qd_quality/latest.ckpt" \
  "${common_qd[@]}" >> "$resume_root/qd_quality.log" 2>&1 &
qd_quality_pid="$!"

env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  OMP_NUM_THREADS="$cpu_threads" MKL_NUM_THREADS="$cpu_threads" \
  OPENBLAS_NUM_THREADS="$cpu_threads" NUMEXPR_NUM_THREADS="$cpu_threads" \
  /home/guoxiangyu/miniconda3/bin/python -u -m methods.qd_detr_gmr.train \
  --variant qd_dual \
  --output_dir artifacts/strict_bsz32/qd_detr/seed2023/qd_dual \
  --resume "$repo_root/artifacts/strict_bsz32/qd_detr/seed2023/qd_dual/latest.ckpt" \
  "${common_qd[@]}" >> "$resume_root/qd_dual.log" 2>&1 &
qd_dual_pid="$!"

FLASH_ALLOW_EXISTING=1 FLASH_EVAL_EPOCH=5 FLASH_PATIENCE=80 \
FLASH_RESUME_CHECKPOINT="$repo_root/artifacts/flash_vtg_supplement/seed2023_bsz128/flash_vtg_plain/model_latest.ckpt" \
  bash "$repo_root/scripts/run_flash_vtg_strict.sh" \
    plain 0 2023 128 \
    "$repo_root/artifacts/flash_vtg_supplement/seed2023_bsz128/flash_vtg_plain" \
    >> "$resume_root/flash_plain.log" 2>&1 &
flash_plain_pid="$!"

FLASH_ALLOW_EXISTING=1 FLASH_EVAL_EPOCH=5 FLASH_PATIENCE=80 \
FLASH_RESUME_CHECKPOINT="$repo_root/artifacts/flash_vtg_supplement/seed2023_bsz128/flash_vtg_gmr/model_latest.ckpt" \
  bash "$repo_root/scripts/run_flash_vtg_strict.sh" \
    gmr 1 2023 128 \
    "$repo_root/artifacts/flash_vtg_supplement/seed2023_bsz128/flash_vtg_gmr" \
    >> "$resume_root/flash_gmr.log" 2>&1 &
flash_gmr_pid="$!"

{
  printf 'qd_quality=%s\n' "$qd_quality_pid"
  printf 'qd_dual=%s\n' "$qd_dual_pid"
  printf 'flash_plain=%s\n' "$flash_plain_pid"
  printf 'flash_gmr=%s\n' "$flash_gmr_pid"
} > "$resume_root/children.pid"

failed=0
wait "$qd_quality_pid" || failed=1
wait "$qd_dual_pid" || failed=1
wait "$flash_plain_pid" || failed=1
wait "$flash_gmr_pid" || failed=1
if [[ "$failed" -ne 0 ]]; then
  printf 'failed:interval5_resume\n' > "$queue_root/queue.status"
  exit 1
fi

printf 'interval5_wave1_completed\n' > "$queue_root/queue.status"
bash "$repo_root/scripts/run_supplementary_after_wave1.sh"
