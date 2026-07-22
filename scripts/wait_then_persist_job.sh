#!/usr/bin/env bash
set -euo pipefail

job_name="${1:?usage: $0 <job-name> <codex-pid> <original-pid> <command> [args...]}"
codex_pid="${2:?missing codex pid}"
original_pid="${3:?missing original pid}"
shift 3

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
log_dir="artifacts/persistent_handoff"
mkdir -p "$log_dir"
exec >> "$log_dir/${job_name}.log" 2>&1

echo "$(date '+%F %T %Z') armed codex_pid=$codex_pid original_pid=$original_pid command=$*"
while kill -0 "$codex_pid" 2>/dev/null; do
  sleep 5
done

# Give ssh/codex session cleanup time to either leave detached jobs alive or
# terminate the whole original process tree before deciding whether to resume.
sleep 20
if kill -0 "$original_pid" 2>/dev/null; then
  echo "$(date '+%F %T %Z') original process survived terminal closure; no duplicate launch"
  while kill -0 "$original_pid" 2>/dev/null; do
    sleep 30
  done
  echo "$(date '+%F %T %Z') surviving original process finished"
  exit 0
fi

echo "$(date '+%F %T %Z') original process absent; starting persistent recovery"
exec "$@"
