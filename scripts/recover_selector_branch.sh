#!/usr/bin/env bash
set -euo pipefail

root="${1:?usage: $0 <result-root> <gpu-id> <seed>}"
gpu_id="${2:?missing gpu id}"
seed="${3:?missing seed}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
export SELECTOR_ROOT="$root"
export SELECTOR_SEED="$seed"

summary="$root/stage4_5_selection/learned_selector_ablation_summary.json"
if [[ -f "$summary" ]]; then
  echo "selector branch already complete: $summary"
  exit 0
fi

if [[ ! -f "$root/stage2_zero/best.ckpt" ]]; then
  echo "missing Stage-2 best checkpoint: $root/stage2_zero/best.ckpt" >&2
  exit 3
fi

if [[ ! -f "$root/stage2_gate_calibration.json" ]]; then
  bash scripts/run_learned_selector_stage.sh stage2_calibrate "$gpu_id"
fi
if [[ ! -f "$root/stage3_pairwise/best.ckpt" ]]; then
  bash scripts/run_learned_selector_stage.sh stage3_pairwise "$gpu_id"
fi
if [[ ! -f "$summary" ]]; then
  bash scripts/run_learned_selector_stage.sh stage4_5_selection "$gpu_id"
fi
