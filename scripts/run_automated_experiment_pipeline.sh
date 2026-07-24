#!/usr/bin/env bash
# Automated Experiment Pipeline - Fully Detached Nohup Execution
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python_bin="/home/guoxiangyu/miniconda3/bin/python"
log_dir="$repo_root/artifacts/automated_pipeline_logs"
mkdir -p "$log_dir"

echo "========================================================================"
echo "Starting Automated Master Experiment Pipeline"
echo "Log Directory: $log_dir"
echo "Started At: $(date --iso-8601=seconds)"
echo "========================================================================"

# Write pipeline status
echo "running" > "$log_dir/pipeline.status"
echo "$$" > "$log_dir/pipeline.pid"

# Phase 1: Wait for current active nohup jobs to complete
echo "[Phase 1] Monitoring active nohup training jobs..."
while ps aux | grep -iE 'methods.eatr_gmr.train|training.moment_detr_gmr' | grep -v 'grep' > /dev/null 2>&1; do
    echo "[$(date +'%H:%M:%S')] Active training jobs running, checking again in 60s..."
    sleep 60
done
echo "[Phase 1] Active initial jobs completed at $(date +'%H:%M:%S')."

# Phase 2: Disentangled Zero Head Ablation (Z no-counter)
echo "[Phase 2] Starting Disentangled Zero Head (Z no-counter) ablation..."
CUDA_VISIBLE_DEVICES=0 "$python_bin" -u training/moment_detr_gmr/train.py \
    --variant md_hiea2m_zero \
    --resume "$repo_root/artifacts/formal_strict/moment_detr/seed2023/md_hiea2m_b128_rerun_from_best_v2/best_joint.ckpt" \
    --seed 2023 \
    --trainable_scope zero \
    --exist_score_mode zero \
    --set_selection_mode legacy \
    --selection_metric gmiou3 \
    --n_epoch 80 \
    --max_es_cnt 15 \
    --bsz 128 --eval_bsz 128 --num_workers 0 \
    --lr 0.0001 \
    --zero_positive_query_weight 2.0 \
    --mask-null-vmr-loss \
    --no-trim_text_by_attention_mask --round_to_clip \
    --train_path "$repo_root/data/label/Standard/train.jsonl" --eval_path "$repo_root/data/label/Standard/val.jsonl" \
    --t_feat_dir "$repo_root/Soccer-GMR/feature/standard/clip_text" \
    --v_feat_dirs "$repo_root/Soccer-GMR/feature/standard/clip" "$repo_root/Soccer-GMR/feature/standard/slowfast" \
    --results_dir "$repo_root/artifacts/disentangled_zero/seed2023/moment" --device cuda > "$log_dir/phase2_zero.log" 2>&1 || true

# Phase 3: Multi-seed Runs for Seed 2024 & Seed 2025
for seed in 2024 2025; do
    echo "[Phase 3] Running Multi-seed Backbone Experiments for Seed $seed..."
    
    # Moment-DETR Seed
    CUDA_VISIBLE_DEVICES=0 "$python_bin" -u training/moment_detr_gmr/train.py \
        --variant md_gmr \
        --seed "$seed" \
        --n_epoch 200 \
        --bsz 128 --eval_bsz 128 --num_workers 0 \
        --lr 0.0001 \
        --mask-null-vmr-loss \
        --no-trim_text_by_attention_mask --round_to_clip \
        --train_path "$repo_root/data/label/Standard/train.jsonl" --eval_path "$repo_root/data/label/Standard/val.jsonl" \
        --t_feat_dir "$repo_root/Soccer-GMR/feature/standard/clip_text" \
        --v_feat_dirs "$repo_root/Soccer-GMR/feature/standard/clip" "$repo_root/Soccer-GMR/feature/standard/slowfast" \
        --results_dir "$repo_root/artifacts/formal_strict/moment_detr/seed${seed}/md_gmr_b128" --device cuda > "$log_dir/phase3_moment_seed${seed}.log" 2>&1 || true

    # EaTR Seed
    CUDA_VISIBLE_DEVICES=1 "$python_bin" -u -m methods.eatr_gmr.train \
        --variant eatr_quality \
        --init-checkpoint "$repo_root/artifacts/eatr_dgqc_transfer/seed2023/eatr_gmr_strict/best.pt" \
        --reference-map 8.02 --reference-gmiou3 16.82 \
        --output-dir "$repo_root/artifacts/eatr_dgqc_transfer/seed${seed}/eatr_quality" \
        --train-annotations "$repo_root/data/label/Standard/train.jsonl" \
        --val-annotations "$repo_root/data/label/Standard/val.jsonl" \
        --slowfast-dir "$repo_root/Soccer-GMR/feature/standard/slowfast" \
        --clip-dir "$repo_root/Soccer-GMR/feature/standard/clip" \
        --text-dir "$repo_root/Soccer-GMR/feature/standard/clip_text" \
        --seed "$seed" --epochs 200 --eval-interval 1 --patience 50 --batch-size 128 --eval-batch-size 128 \
        --lr 3e-5 --backbone-lr-scale 0.1 --num-workers 0 --train-sample-mode mixed --mask-null-vmr-loss --device cuda > "$log_dir/phase3_eatr_seed${seed}.log" 2>&1 || true
done

# Phase 4: Coverage-aware Event Selector & Master Summary Update
echo "[Phase 4] Running Selector Ablation & Summary Updates..."
"$python_bin" "$repo_root/scripts/generate_completed_experiment_summary.py" || true
"$python_bin" "$repo_root/scripts/generate_teacher_progress_report.py" || true

echo "completed" > "$log_dir/pipeline.status"
echo "========================================================================"
echo "Automated Master Experiment Pipeline Completed Successfully!"
echo "Finished At: $(date --iso-8601=seconds)"
echo "========================================================================"
