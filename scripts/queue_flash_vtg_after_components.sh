#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

mode="${1:-start}"
queue_root="${FLASH_QUEUE_ROOT:-$repo_root/artifacts/flash_vtg_supplement/seed${FLASH_SEED:-2023}/queue}"

start_detached() {
  mkdir -p "$queue_root"
  exec bash "$repo_root/scripts/launch_nohup_job.sh" \
    "$queue_root" 0 \
    bash "$repo_root/scripts/queue_flash_vtg_after_components.sh" worker
}

run_worker() {
  local seed="${FLASH_SEED:-2023}"
  local batch_size="${FLASH_BSZ:-8}"
  local plain_gpu="${FLASH_PLAIN_GPU:-0}"
  local gmr_gpu="${FLASH_GMR_GPU:-1}"
  local poll_seconds="${FLASH_POLL_SECONDS:-30}"
  local run_root="${FLASH_RUN_ROOT:-$repo_root/artifacts/flash_vtg_supplement/seed${seed}}"
  local blocker_pattern='methods\.(qd_detr_gmr|cg_detr_gmr)\.train'
  local status_file="$queue_root/queue.status"
  local wait_file="$queue_root/waited_processes.txt"

  for value_name in batch_size poll_seconds; do
    local value="${!value_name}"
    if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
      echo "$value_name must be a positive integer: $value" >&2
      exit 2
    fi
  done
  for value_name in seed plain_gpu gmr_gpu; do
    local value="${!value_name}"
    if [[ ! "$value" =~ ^[0-9]+$ ]]; then
      echo "$value_name must be a non-negative integer: $value" >&2
      exit 2
    fi
  done

  mkdir -p "$queue_root" "$run_root"
  printf 'waiting_for_qd_cg\n' > "$status_file"
  : > "$wait_file"

  local blockers=()
  if [[ -n "${FLASH_WAIT_PIDS:-}" ]]; then
    read -r -a blockers <<< "$FLASH_WAIT_PIDS"
  else
    mapfile -t blockers < <(pgrep -f "$blocker_pattern" || true)
  fi

  local pid
  for pid in "${blockers[@]}"; do
    if [[ ! "$pid" =~ ^[0-9]+$ ]]; then
      echo "ignoring invalid blocker PID: $pid" >&2
      continue
    fi
    if kill -0 "$pid" 2>/dev/null; then
      printf 'pid=%s cmd=' "$pid" >> "$wait_file"
      ps -p "$pid" -o args= >> "$wait_file" 2>/dev/null || printf '<unavailable>\n' >> "$wait_file"
    fi
  done

  echo "Queue captured ${#blockers[@]} current QD/CG training process(es)."
  for pid in "${blockers[@]}"; do
    [[ "$pid" =~ ^[0-9]+$ ]] || continue
    while kill -0 "$pid" 2>/dev/null; do
      state="$(ps -p "$pid" -o stat= 2>/dev/null || true)"
      [[ "$state" == Z* ]] && break
      sleep "$poll_seconds"
    done
    echo "Blocker PID $pid finished."
  done

  printf 'launching_flash_pair\n' > "$status_file"
  local plain_dir="$run_root/flash_vtg_plain"
  local gmr_dir="$run_root/flash_vtg_gmr"

  bash "$repo_root/scripts/run_flash_vtg_strict.sh" \
    plain "$plain_gpu" "$seed" "$batch_size" "$plain_dir" \
    > "$queue_root/plain.launcher.log" 2>&1 &
  local plain_pid="$!"
  printf '%s\n' "$plain_pid" > "$queue_root/plain.pid"

  bash "$repo_root/scripts/run_flash_vtg_strict.sh" \
    gmr "$gmr_gpu" "$seed" "$batch_size" "$gmr_dir" \
    > "$queue_root/gmr.launcher.log" 2>&1 &
  local gmr_pid="$!"
  printf '%s\n' "$gmr_pid" > "$queue_root/gmr.pid"

  {
    printf 'plain_pid=%s gpu=%s run_dir=%s\n' "$plain_pid" "$plain_gpu" "$plain_dir"
    printf 'gmr_pid=%s gpu=%s run_dir=%s\n' "$gmr_pid" "$gmr_gpu" "$gmr_dir"
    printf 'launched_at=%s\n' "$(date --iso-8601=seconds)"
  } > "$queue_root/flash_launches.txt"
  printf 'running_flash_pair\n' > "$status_file"

  local status=0
  wait "$plain_pid" || status=1
  wait "$gmr_pid" || status=1
  if [[ "$status" -eq 0 ]]; then
    printf 'completed\n' > "$status_file"
    echo "Flash-VTG plain/GMR pair completed successfully."
  else
    printf 'failed\n' > "$status_file"
    echo "At least one Flash-VTG run failed; inspect $queue_root/*.launcher.log." >&2
    return 1
  fi
}

case "$mode" in
  start) start_detached ;;
  worker) run_worker ;;
  *)
    echo "usage: $0 [start|worker]" >&2
    exit 2
    ;;
esac
