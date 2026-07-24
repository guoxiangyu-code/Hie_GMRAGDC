#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

mode="${1:-start}"
seed="${SUPPLEMENT_SEED:-2023}"
batch_size="${FLASH_BSZ:-128}"
queue_root="${SUPPLEMENT_QUEUE_ROOT:-$repo_root/artifacts/supplementary_queue/seed${seed}}"
flash_root="${FLASH_RUN_ROOT:-$repo_root/artifacts/flash_vtg_supplement/seed${seed}_bsz${batch_size}}"
status_file="$queue_root/queue.status"
manifest_file="$queue_root/waves.tsv"

start_detached() {
  mkdir -p "$queue_root"
  exec bash "$repo_root/scripts/launch_nohup_job.sh" \
    "$queue_root" 0 \
    bash "$repo_root/scripts/queue_supplementary_experiments.sh" worker
}

record_wave() {
  printf '%s\t%s\t%s\n' "$(date --iso-8601=seconds)" "$1" "$2" >> "$manifest_file"
}

wait_for_initial_jobs() {
  local wait_file="$queue_root/waited_processes.txt"
  local poll_seconds="${SUPPLEMENT_POLL_SECONDS:-30}"
  local blockers=()
  if [[ -n "${SUPPLEMENT_WAIT_PIDS:-}" ]]; then
    read -r -a blockers <<< "$SUPPLEMENT_WAIT_PIDS"
  else
    mapfile -t blockers < <(
      pgrep -f 'methods\.(qd_detr_gmr|cg_detr_gmr)\.train' || true
    )
  fi

  : > "$wait_file"
  printf 'waiting_for_existing_qd_cg\n' > "$status_file"
  local pid
  for pid in "${blockers[@]}"; do
    [[ "$pid" =~ ^[0-9]+$ ]] || continue
    if kill -0 "$pid" 2>/dev/null; then
      printf 'pid=%s cmd=' "$pid" >> "$wait_file"
      ps -p "$pid" -o args= >> "$wait_file" 2>/dev/null \
        || printf '<unavailable>\n' >> "$wait_file"
    fi
  done
  record_wave "wait" "captured ${#blockers[@]} QD/CG process(es)"

  for pid in "${blockers[@]}"; do
    [[ "$pid" =~ ^[0-9]+$ ]] || continue
    while kill -0 "$pid" 2>/dev/null; do
      state="$(ps -p "$pid" -o stat= 2>/dev/null || true)"
      [[ "$state" == Z* ]] && break
      sleep "$poll_seconds"
    done
  done
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

run_worker() {
  if [[ ! "$seed" =~ ^[0-9]+$ || ! "$batch_size" =~ ^[1-9][0-9]*$ ]]; then
    echo "invalid seed or Flash batch size" >&2
    exit 2
  fi
  mkdir -p "$queue_root" "$flash_root"
  : > "$manifest_file"
  {
    printf 'seed=%s\nflash_batch_size=%s\n' "$seed" "$batch_size"
    printf 'started_at=%s\n' "$(date --iso-8601=seconds)"
    printf 'git_commit=%s\n' "$(git rev-parse HEAD 2>/dev/null || printf unknown)"
  } > "$queue_root/queue_metadata.txt"

  if [[ "${SUPPLEMENT_SKIP_WAIT:-0}" == "1" ]]; then
    printf 'starting_immediately\n' > "$status_file"
    record_wave "wait" "skipped by SUPPLEMENT_SKIP_WAIT=1"
  else
    wait_for_initial_jobs
  fi

  # Wave 1: matched from-scratch Flash plain/GMR controls.
  printf 'wave1_flash_plain_gmr\n' > "$status_file"
  record_wave "wave1" "launch Flash plain/GMR from scratch"
  bash "$repo_root/scripts/run_flash_vtg_strict.sh" \
    plain 0 "$seed" "$batch_size" "$flash_root/flash_vtg_plain" \
    > "$queue_root/wave1_plain.log" 2>&1 &
  wave1_a="$!"
  bash "$repo_root/scripts/run_flash_vtg_strict.sh" \
    gmr 1 "$seed" "$batch_size" "$flash_root/flash_vtg_gmr" \
    > "$queue_root/wave1_gmr.log" 2>&1 &
  wave1_b="$!"
  wait_pair "wave1" "$wave1_a" "$wave1_b"

  # Wave 2: low-cost, frozen-parent single-module screening.
  printf 'wave2_flash_quality_zero_singles\n' > "$status_file"
  record_wave "wave2" "launch Flash Quality and independent Zero"
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

  # Wave 3: decoupled Q+Z (freeze the trained Quality head) plus QD Stage B.
  printf 'wave3_flash_qz_and_qd_stage_b\n' > "$status_file"
  record_wave "wave3" "launch decoupled Flash Q+Z and QD Quality+Dual"
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

  # Wave 4: remaining matched-parent Q+D combinations.
  printf 'wave4_cg_moment_stage_b\n' > "$status_file"
  record_wave "wave4" "launch CG and Moment Quality+Dual"
  bash "$repo_root/scripts/run_stage_b_quality_dual.sh" cg 0 \
    > "$queue_root/wave4_cg_quality_phrase.log" 2>&1 &
  wave4_a="$!"
  bash "$repo_root/scripts/run_stage_b_quality_dual.sh" moment 1 \
    > "$queue_root/wave4_moment_quality_dual.log" 2>&1 &
  wave4_b="$!"
  wait_pair "wave4" "$wave4_a" "$wave4_b"

  printf 'wave5_eatr_stage_b\n' > "$status_file"
  record_wave "wave5" "launch EaTR Quality+Dual"
  bash "$repo_root/scripts/run_stage_b_quality_dual.sh" eatr 0 \
    > "$queue_root/wave5_eatr_quality_dual.log" 2>&1
  record_wave "wave5" "completed"

  printf 'completed\n' > "$status_file"
  record_wave "queue" "all scheduled experiments completed"
}

case "$mode" in
  start) start_detached ;;
  worker) run_worker ;;
  *)
    echo "usage: $0 [start|worker]" >&2
    exit 2
    ;;
esac
