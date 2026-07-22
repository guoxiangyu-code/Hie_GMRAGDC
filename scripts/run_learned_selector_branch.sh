#!/usr/bin/env bash
set -euo pipefail

root="${1:?usage: $0 <result-root> <gpu-id> <seed> <zero-positive-weight>}"
gpu_id="${2:?missing gpu id}"
seed="${3:?missing seed}"
zero_positive_weight="${4:?missing zero positive weight}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

export SELECTOR_ROOT="$root"
export SELECTOR_SEED="$seed"
export ZERO_POSITIVE_WEIGHT="$zero_positive_weight"

bash scripts/run_learned_selector_stage.sh stage2_zero "$gpu_id"
bash scripts/run_learned_selector_stage.sh stage2_calibrate "$gpu_id"
bash scripts/run_learned_selector_stage.sh stage3_pairwise "$gpu_id"
bash scripts/run_learned_selector_stage.sh stage4_5_selection "$gpu_id"
