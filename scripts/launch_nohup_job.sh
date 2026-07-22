#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 RUN_DIR GPU_ID COMMAND [ARG ...]" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
run_dir="$1"
gpu_id="$2"
shift 2

if [[ "$run_dir" != /* ]]; then
  run_dir="$repo_root/$run_dir"
fi
run_dir="$(realpath -m "$run_dir")"

case "$run_dir" in
  "$repo_root"/*) ;;
  *)
    echo "RUN_DIR must be inside $repo_root: $run_dir" >&2
    exit 2
    ;;
esac

if [[ ! "$gpu_id" =~ ^[0-9]+$ ]]; then
  echo "GPU_ID must be a non-negative integer: $gpu_id" >&2
  exit 2
fi

mkdir -p "$run_dir"
pid_file="$run_dir/pid"
if [[ -s "$pid_file" ]]; then
  old_pid="$(<"$pid_file")"
  if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "Refusing to launch over live PID $old_pid in $run_dir" >&2
    exit 3
  fi
fi

command_file="$run_dir/launch_command.txt"
metadata_file="$run_dir/launch_metadata.txt"
status_file="$run_dir/git_status_at_launch.txt"
stdout_file="$run_dir/nohup.log"

{
  printf 'CUDA_VISIBLE_DEVICES=%q ' "$gpu_id"
  printf 'PYTHONUNBUFFERED=1 '
  printf 'PYTORCH_CUDA_ALLOC_CONF=%q ' 'expandable_segments:True'
  printf '%q ' "$@"
  printf '\n'
} > "$command_file"

{
  printf 'launched_at=%s\n' "$(date --iso-8601=seconds)"
  printf 'repo_root=%s\n' "$repo_root"
  printf 'run_dir=%s\n' "$run_dir"
  printf 'gpu_id=%s\n' "$gpu_id"
  printf 'hostname=%s\n' "$(hostname)"
  printf 'git_commit=%s\n' "$(git -C "$repo_root" rev-parse HEAD 2>/dev/null || printf unknown)"
  printf 'stdout_log=%s\n' "$stdout_file"
  printf 'command_file=%s\n' "$command_file"
} > "$metadata_file"
git -C "$repo_root" status --short > "$status_file"

# Start the job in its own session as well as ignoring SIGHUP.  A plain
# backgrounded `nohup` process remains in the caller's process group, which
# lets session cleanup terminate long-running training after this launcher
# returns.
nohup setsid env \
  CUDA_VISIBLE_DEVICES="$gpu_id" \
  PYTHONUNBUFFERED=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$@" > "$stdout_file" 2>&1 < /dev/null &
pid="$!"
printf '%s\n' "$pid" > "$pid_file"
printf 'pid=%s\n' "$pid" >> "$metadata_file"

# Catch immediate import/argument failures while still returning promptly for
# genuine long-running jobs.
sleep 2
if ! kill -0 "$pid" 2>/dev/null; then
  echo "Job exited during startup (PID $pid); inspect $stdout_file" >&2
  exit 4
fi

printf 'Started PID %s on physical GPU %s\n' "$pid" "$gpu_id"
printf 'Log: %s\n' "$stdout_file"
