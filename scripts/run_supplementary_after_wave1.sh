#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

seed="${SUPPLEMENT_SEED:-2023}"
batch_size="${FLASH_BSZ:-128}"
queue_root="${SUPPLEMENT_QUEUE_ROOT:-$repo_root/artifacts/supplementary_queue/seed${seed}}"
flash_root="${FLASH_RUN_ROOT:-$repo_root/artifacts/flash_vtg_supplement/seed${seed}_bsz${batch_size}}"
status_file="$queue_root/queue.status"
manifest_file="$queue_root/waves.tsv"
parallel_active="$queue_root/parallel_prefetch.active"
parallel_completed="$queue_root/parallel_prefetch.completed"
parallel_failed="$queue_root/parallel_prefetch.failed"

if [[ -e "$parallel_active" ]]; then
  printf 'waiting_for_parallel_prefetch\n' > "$status_file"
  while [[ -e "$parallel_active" ]]; do
    sleep 30
  done
fi
if [[ -e "$parallel_completed" ]]; then
  printf 'completed\n' > "$status_file"
  printf '%s\tqueue\tparallel prefetch already completed all remaining waves\n' \
    "$(date --iso-8601=seconds)" >> "$manifest_file"
  exit 0
fi
if [[ -e "$parallel_failed" ]]; then
  printf 'failed:parallel_prefetch\n' > "$status_file"
  exit 1
fi

record_wave() {
  printf '%s\t%s\t%s\n' "$(date --iso-8601=seconds)" "$1" "$2" >> "$manifest_file"
}

wait_pair() {
  local wave="$1"
  local first_pid="$2"
  local second_pid="$3"
  local failed=0
  wait "$first_pid" || failed=1
  wait "$second_pid" || failed=1
  if [[ "$failed" -ne 0 ]]; then
    printf 'failed:%s\n' "$wave" > "$status_file"
    record_wave "$wave" "failed"
    return 1
  fi
  record_wave "$wave" "completed"
}

mkdir -p "$queue_root" "$flash_root"

printf 'wave2_flash_quality_zero_singles\n' > "$status_file"
record_wave "wave2" "launch Flash Quality and independent Zero; eval every 5 epochs"
FLASH_EPOCHS=120 FLASH_PATIENCE=30 FLASH_EVAL_EPOCH=5 \
  FLASH_SELECTION_METRIC=mAP \
  bash "$repo_root/scripts/run_flash_vtg_strict.sh" \
    gmr_quality 0 "$seed" "$batch_size" "$flash_root/flash_vtg_gmr_quality" \
    > "$queue_root/wave2_quality.log" 2>&1 &
wave2_a="$!"
FLASH_EPOCHS=120 FLASH_PATIENCE=30 FLASH_EVAL_EPOCH=5 \
  FLASH_SELECTION_METRIC=joint \
  FLASH_REFERENCE_MAP=26.01 FLASH_REFERENCE_GMIOU3=33.93 \
  bash "$repo_root/scripts/run_flash_vtg_strict.sh" \
    gmr_zero 1 "$seed" "$batch_size" "$flash_root/flash_vtg_gmr_zero" \
    > "$queue_root/wave2_zero.log" 2>&1 &
wave2_b="$!"
wait_pair "wave2" "$wave2_a" "$wave2_b"

printf 'wave3_flash_qz_and_qd_stage_b\n' > "$status_file"
record_wave "wave3" "launch Flash Q+Z and QD Q+D; eval every 5 epochs"
FLASH_EPOCHS=120 FLASH_PATIENCE=30 FLASH_EVAL_EPOCH=5 \
  FLASH_SELECTION_METRIC=joint \
  FLASH_REFERENCE_MAP=26.01 FLASH_REFERENCE_GMIOU3=33.93 \
  FLASH_INIT_CHECKPOINT="$flash_root/flash_vtg_gmr_quality/model_best_map.ckpt" \
  FLASH_FREEZE_QUALITY=1 \
  bash "$repo_root/scripts/run_flash_vtg_strict.sh" \
    gmr_quality_zero 0 "$seed" "$batch_size" \
    "$flash_root/flash_vtg_gmr_quality_zero" \
    > "$queue_root/wave3_quality_zero.log" 2>&1 &
wave3_a="$!"
bash "$repo_root/scripts/run_stage_b_quality_dual.sh" qd 1 \
  > "$queue_root/wave3_qd_quality_dual.log" 2>&1 &
wave3_b="$!"
wait_pair "wave3" "$wave3_a" "$wave3_b"

printf 'wave4_moment_stage_b_cg_skipped\n' > "$status_file"
record_wave "wave4" "skip CG; launch Moment Q+D; eval every 5 epochs"
bash "$repo_root/scripts/run_stage_b_quality_dual.sh" moment 1 \
  > "$queue_root/wave4_moment_quality_dual.log" 2>&1
record_wave "wave4" "completed; CG skipped"

printf 'wave5_eatr_stage_b\n' > "$status_file"
record_wave "wave5" "launch EaTR Q+D; eval every 5 epochs"
bash "$repo_root/scripts/run_stage_b_quality_dual.sh" eatr 0 \
  > "$queue_root/wave5_eatr_quality_dual.log" 2>&1
record_wave "wave5" "completed"

printf 'completed\n' > "$status_file"
record_wave "queue" "all scheduled interval-5 experiments completed"
