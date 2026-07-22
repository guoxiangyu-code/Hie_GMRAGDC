#!/usr/bin/env bash
set -euo pipefail

training_pid="${1:?usage: $0 <training-pid> <result-root> <gpu-id>}"
root="${2:?missing result root}"
gpu_id="${3:?missing gpu id}"

if [[ ! "$training_pid" =~ ^[0-9]+$ ]]; then
  echo "invalid training pid: $training_pid" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

while ps -p "$training_pid" -o args= 2>/dev/null | grep -Fq -- "--results_dir $root/stage2_zero"; do
  sleep 30
done

if [[ ! -f "$root/stage2_zero/best.ckpt" ]]; then
  echo "stage-2 process ended without a best checkpoint: $root/stage2_zero/best.ckpt" >&2
  exit 3
fi

export SELECTOR_ROOT="$root"
bash scripts/run_learned_selector_stage.sh stage2_calibrate "$gpu_id"
bash scripts/run_learned_selector_stage.sh stage3_pairwise "$gpu_id"
bash scripts/run_learned_selector_stage.sh stage4_5_selection "$gpu_id"
