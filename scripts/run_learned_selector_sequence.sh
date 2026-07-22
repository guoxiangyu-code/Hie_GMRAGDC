#!/usr/bin/env bash
set -euo pipefail

gpu_id="${1:-0}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

# Foreground execution is intentional: no nohup or screen is created here.
bash scripts/run_learned_selector_stage.sh stage1_geometry "$gpu_id"
bash scripts/run_learned_selector_stage.sh stage2_zero "$gpu_id"
bash scripts/run_learned_selector_stage.sh stage2_calibrate "$gpu_id"
bash scripts/run_learned_selector_stage.sh stage3_pairwise "$gpu_id"
bash scripts/run_learned_selector_stage.sh stage4_5_selection "$gpu_id"
