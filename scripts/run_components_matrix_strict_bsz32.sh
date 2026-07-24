#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

backbone="${1:?usage: $0 <qd|cg>}"
gpu_id="${2:-0}"

if [[ "$backbone" == "qd" ]]; then
    variants=("qd_quality" "qd_dual" "qd_counter" "qd_hiea2m")
    parent_ckpt="artifacts/strict_bsz32/qd_detr/seed2023/qd_detr_gmr/best_joint.ckpt"
    ref_json="artifacts/strict_bsz32/qd_detr/seed2023/qd_detr_gmr/best_joint_val_metrics.json"
    module_name="methods.qd_detr_gmr.train"
    out_base="artifacts/strict_bsz32/qd_detr/seed2023"
elif [[ "$backbone" == "cg" ]]; then
    variants=("cg_quality" "cg_phrase" "cg_counter" "cg_hiea2m")
    parent_ckpt="artifacts/strict_bsz32/cg_detr/seed2023/cg_detr_gmr/best_joint.ckpt"
    ref_json="artifacts/strict_bsz32/cg_detr/seed2023/cg_detr_gmr/best_joint_val_metrics.json"
    module_name="methods.cg_detr_gmr.train"
    out_base="artifacts/strict_bsz32/cg_detr/seed2023"
else
    echo "Unknown backbone: $backbone" >&2
    exit 1
fi

if [[ ! -f "$parent_ckpt" ]]; then
    echo "Missing parent checkpoint: $parent_ckpt" >&2
    exit 1
fi

if [[ ! -f "$ref_json" ]]; then
    echo "Missing reference metrics JSON: $ref_json" >&2
    exit 1
fi

ref_map=$(python3 -c "import json; b=json.load(open('$ref_json'))['brief']; print(b['mAP'])")
ref_gmiou3=$(python3 -c "import json; b=json.load(open('$ref_json'))['brief']; print(b['G-mIoU@3'])")

echo "Starting PARALLEL $backbone matrix. Reference metrics from GMR: mAP=$ref_map, G-mIoU@3=$ref_gmiou3"

pids=()
for var in "${variants[@]}"; do
    out_dir="$out_base/$var"
    mkdir -p "$out_dir"
    echo "Started variant $var on GPU $gpu_id in background (PID will follow)..."
    
    CUDA_VISIBLE_DEVICES="$gpu_id" python -u -m "$module_name" \
        --variant "$var" \
        --seed 2023 --epochs 200 --batch_size 32 --eval_bsz 32 \
        --patience 50 --eval_interval 1 --num_workers 0 \
        --lr 3e-5 --backbone_lr_scale 0.1 \
        --train_sample_mode mixed --map_num_workers 1 \
        --reference_map "$ref_map" --reference_gmiou3 "$ref_gmiou3" \
        --train_annotation data/label/Standard/train.jsonl \
        --eval_annotation data/label/Standard/val.jsonl \
        --video_feature_dirs Soccer-GMR/feature/standard/clip Soccer-GMR/feature/standard/slowfast \
        --text_feature_dir Soccer-GMR/feature/standard/clip_text \
        --output_dir "$out_dir" --device cuda --round_to_clip \
        --init_checkpoint "$parent_ckpt" \
        --mask-null-vmr-loss > "$out_dir/stdout.log" 2>&1 &
        
    pid=$!
    pids+=("$pid")
    echo "  -> PID $pid assigned to $var"
done

echo "Waiting for all parallel jobs to complete..."
status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done

if [[ $status -ne 0 ]]; then
    echo "Some variants failed. Check logs in $out_base." >&2
    exit 1
fi
echo "All $backbone variants completed successfully."
