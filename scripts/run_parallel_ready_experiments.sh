#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

seed="${SUPPLEMENT_SEED:-2023}"
batch_size="${FLASH_BSZ:-128}"
cpu_threads="${GMR_CPU_THREADS:-6}"
queue_root="$repo_root/artifacts/supplementary_queue/seed${seed}"
flash_root="$repo_root/artifacts/flash_vtg_supplement/seed${seed}_bsz${batch_size}"
parallel_root="$queue_root/parallel_ready"
active_marker="$queue_root/parallel_prefetch.active"
completed_marker="$queue_root/parallel_prefetch.completed"
failed_marker="$queue_root/parallel_prefetch.failed"

if [[ -e "$active_marker" || -e "$completed_marker" ]]; then
  echo "parallel supplementary batch already active or completed" >&2
  exit 2
fi
mkdir -p "$parallel_root" "$flash_root"
printf 'pid=%s started=%s\n' "$$" "$(date --iso-8601=seconds)" > "$active_marker"
printf 'parallel_prefetch_flash_q_z_and_stage_b\n' > "$queue_root/queue.status"

finish() {
  exit_code="$?"
  if [[ "$exit_code" -ne 0 ]]; then
    mv -f "$active_marker" "$failed_marker"
    printf 'failed:parallel_prefetch\n' > "$queue_root/queue.status"
  fi
}
trap finish EXIT

FLASH_EPOCHS=120 FLASH_PATIENCE=30 FLASH_EVAL_EPOCH=5 \
FLASH_SELECTION_METRIC=mAP GMR_CPU_THREADS="$cpu_threads" \
  bash "$repo_root/scripts/run_flash_vtg_strict.sh" \
    gmr_quality 0 "$seed" "$batch_size" "$flash_root/flash_vtg_gmr_quality" \
    > "$parallel_root/flash_quality.log" 2>&1 &
flash_quality_pid="$!"

FLASH_EPOCHS=120 FLASH_PATIENCE=30 FLASH_EVAL_EPOCH=5 \
FLASH_SELECTION_METRIC=joint \
FLASH_REFERENCE_MAP=26.01 FLASH_REFERENCE_GMIOU3=33.93 \
GMR_CPU_THREADS="$cpu_threads" \
  bash "$repo_root/scripts/run_flash_vtg_strict.sh" \
    gmr_zero 1 "$seed" "$batch_size" "$flash_root/flash_vtg_gmr_zero" \
    > "$parallel_root/flash_zero.log" 2>&1 &
flash_zero_pid="$!"

GMR_CPU_THREADS="$cpu_threads" \
  bash "$repo_root/scripts/run_stage_b_quality_dual.sh" qd 1 \
    > "$parallel_root/qd_quality_dual.log" 2>&1 &
qd_pid="$!"

GMR_CPU_THREADS="$cpu_threads" \
  bash "$repo_root/scripts/run_stage_b_quality_dual.sh" moment 0 \
    > "$parallel_root/moment_quality_dual.log" 2>&1 &
moment_pid="$!"

GMR_CPU_THREADS="$cpu_threads" \
  bash "$repo_root/scripts/run_stage_b_quality_dual.sh" eatr 1 \
    > "$parallel_root/eatr_quality_dual.log" 2>&1 &
eatr_pid="$!"

{
  printf 'flash_quality=%s\nflash_zero=%s\n' "$flash_quality_pid" "$flash_zero_pid"
  printf 'qd_quality_dual=%s\nmoment_quality_dual=%s\neatr_quality_dual=%s\n' \
    "$qd_pid" "$moment_pid" "$eatr_pid"
} > "$parallel_root/children.pid"

# Q+Z depends only on the trained Quality checkpoint; start it immediately
# after Q finishes while the other independent jobs continue.
wait "$flash_quality_pid"
quality_checkpoint="$flash_root/flash_vtg_gmr_quality/model_best_map.ckpt"
if [[ ! -s "$quality_checkpoint" ]]; then
  echo "Flash Quality completed without model_best_map.ckpt" >&2
  exit 3
fi
FLASH_EPOCHS=120 FLASH_PATIENCE=30 FLASH_EVAL_EPOCH=5 \
FLASH_SELECTION_METRIC=joint \
FLASH_REFERENCE_MAP=26.01 FLASH_REFERENCE_GMIOU3=33.93 \
FLASH_INIT_CHECKPOINT="$quality_checkpoint" FLASH_FREEZE_QUALITY=1 \
GMR_CPU_THREADS="$cpu_threads" \
  bash "$repo_root/scripts/run_flash_vtg_strict.sh" \
    gmr_quality_zero 0 "$seed" "$batch_size" \
    "$flash_root/flash_vtg_gmr_quality_zero" \
    > "$parallel_root/flash_quality_zero.log" 2>&1 &
flash_qz_pid="$!"
printf 'flash_quality_zero=%s\n' "$flash_qz_pid" >> "$parallel_root/children.pid"

failed=0
wait "$flash_zero_pid" || failed=1
wait "$qd_pid" || failed=1
wait "$moment_pid" || failed=1
wait "$eatr_pid" || failed=1
wait "$flash_qz_pid" || failed=1
if [[ "$failed" -ne 0 ]]; then
  exit 1
fi

mv -f "$active_marker" "$completed_marker"
printf 'parallel_prefetch_completed\n' > "$queue_root/queue.status"
printf '%s\tparallel\tcompleted Flash Q/Z/Q+Z and QD/Moment/EaTR Stage-B\n' \
  "$(date --iso-8601=seconds)" >> "$queue_root/waves.tsv"
trap - EXIT
