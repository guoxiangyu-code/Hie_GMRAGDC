#!/usr/bin/env bash
# Master Launcher for Strictly Matched Ablation (B / Q / Z / Q+Z / Q+Z+P) Across 3 Backbones
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python_bin="/home/guoxiangyu/miniconda3/bin/python"
output_root="$repo_root/artifacts/strictly_matched_ablation/seed2023"
mkdir -p "$output_root"

echo "========================================================================"
echo "Starting Strictly Matched Ablation Suite (B / Q / Z / Q+Z / Q+Z+P)"
echo "Output Directory: $output_root"
echo "Started At: $(date --iso-8601=seconds)"
echo "========================================================================"

# Write metadata
{
  echo "started_at=$(date --iso-8601=seconds)"
  echo "seed=2023"
  echo "backbones=moment_detr,eatr,flash_vtg"
  echo "steps=B,Q,Z,Q+Z,Q+Z+P"
} > "$output_root/master_metadata.txt"

# 1. Moment-DETR Execution (GPU 0)
echo "[GPU 0] Launching Moment-DETR Matched Suite..."
bash "$repo_root/scripts/run_learned_selector_stage.sh" 0 > "$output_root/moment_detr_suite.log" 2>&1 &
moment_pid=$!

# 2. EaTR & Flash-VTG Execution (GPU 1)
echo "[GPU 1] Launching EaTR Matched Transfer Suite..."
bash "$repo_root/scripts/run_eatr_dgqc_transfer.sh" children 1 > "$output_root/eatr_suite.log" 2>&1 &
eatr_pid=$!

echo "[Master] Jobs launched successfully."
echo "  Moment-DETR PID: $moment_pid (logs: $output_root/moment_detr_suite.log)"
echo "  EaTR PID: $eatr_pid (logs: $output_root/eatr_suite.log)"
echo "========================================================================"
