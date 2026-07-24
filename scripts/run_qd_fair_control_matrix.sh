#!/usr/bin/env bash
# Matched QD ablation: continued GMR control vs Quality vs Dual vs Quality+Dual.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python_bin="/home/guoxiangyu/miniconda3/bin/python"
seed="${QD_FAIR_SEED:-2023}"
cpu_threads="${GMR_CPU_THREADS:-4}"
output_root="${QD_FAIR_OUTPUT_ROOT:-$repo_root/artifacts/qd_fair_ablation/seed${seed}_bsz32}"
parent="$repo_root/artifacts/strict_bsz32/qd_detr/seed2023/qd_detr_gmr/best_joint.ckpt"
reference_json="$repo_root/artifacts/strict_bsz32/qd_detr/seed2023/qd_detr_gmr/best_joint_val_metrics.json"

if [[ ! "$cpu_threads" =~ ^[4-8]$ ]]; then
  echo "GMR_CPU_THREADS must be between 4 and 8: $cpu_threads" >&2
  exit 2
fi
if [[ ! -f "$parent" || ! -f "$reference_json" ]]; then
  echo "missing matched QD parent checkpoint or metrics" >&2
  exit 3
fi

reference_map="$("$python_bin" -c "import json; print(json.load(open('$reference_json'))['brief']['mAP'])")"
reference_gmiou3="$("$python_bin" -c "import json; print(json.load(open('$reference_json'))['brief']['G-mIoU@3'])")"
mkdir -p "$output_root"

variants=(qd_detr_gmr qd_quality qd_dual qd_quality_dual)
labels=(continued_control quality dual quality_dual)
gpus=(0 0 1 1)

# Refuse a partial accidental overwrite; resumption should be implemented as a
# separate explicit operation after inspecting the failed job.
for label in "${labels[@]}"; do
  run_dir="$output_root/$label"
  if [[ -e "$run_dir/train_log.jsonl" || -e "$run_dir/latest.ckpt" ]]; then
    echo "refusing to overwrite existing fair-ablation run: $run_dir" >&2
    exit 4
  fi
done

{
  printf 'started_at=%s\nseed=%s\nbatch_size=32\neval_interval=5\npatience=10\n' \
    "$(date --iso-8601=seconds)" "$seed"
  printf 'parent=%s\nreference_map=%s\nreference_gmiou3=%s\n' \
    "$parent" "$reference_map" "$reference_gmiou3"
  printf 'comparison=continued_control,quality,dual,quality_dual\n'
} > "$output_root/matrix_metadata.txt"
printf 'running\n' > "$output_root/matrix.status"
printf '%s\n' "$$" > "$output_root/matrix.pid"

pids=()
for index in "${!variants[@]}"; do
  variant="${variants[$index]}"
  label="${labels[$index]}"
  gpu="${gpus[$index]}"
  run_dir="$output_root/$label"
  mkdir -p "$run_dir"

  command=(
    "$python_bin" -u -m methods.qd_detr_gmr.train
    --variant "$variant"
    --seed "$seed"
    --epochs 200
    --batch_size 32
    --eval_bsz 32
    --patience 10
    --eval_interval 5
    --num_workers 0
    --lr 3e-5
    --backbone_lr_scale 0.1
    --train_sample_mode mixed
    --map_num_workers 1
    --reference_map "$reference_map"
    --reference_gmiou3 "$reference_gmiou3"
    --train_annotation "$repo_root/data/label/Standard/train.jsonl"
    --eval_annotation "$repo_root/data/label/Standard/val.jsonl"
    --video_feature_dirs
      "$repo_root/Soccer-GMR/feature/standard/clip"
      "$repo_root/Soccer-GMR/feature/standard/slowfast"
    --text_feature_dir "$repo_root/Soccer-GMR/feature/standard/clip_text"
    --output_dir "$run_dir"
    --device cuda
    --round_to_clip
    --no-diagnostic_decoders
    --init_checkpoint "$parent"
    --mask-null-vmr-loss
  )

  {
    printf 'label=%s\nvariant=%s\nphysical_gpu=%s\nseed=%s\n' \
      "$label" "$variant" "$gpu" "$seed"
    printf 'parent=%s\nstarted_at=%s\n' "$parent" "$(date --iso-8601=seconds)"
    printf 'CUDA_VISIBLE_DEVICES=%q ' "$gpu"
    printf '%q ' "${command[@]}"
    printf '\n'
  } > "$run_dir/launch_metadata.txt"

  (
    printf 'running\n' > "$run_dir/runner.status"
    set +e
    env CUDA_VISIBLE_DEVICES="$gpu" \
      PYTHONUNBUFFERED=1 \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      OMP_NUM_THREADS="$cpu_threads" \
      MKL_NUM_THREADS="$cpu_threads" \
      OPENBLAS_NUM_THREADS="$cpu_threads" \
      NUMEXPR_NUM_THREADS="$cpu_threads" \
      "${command[@]}" > "$run_dir/stdout.log" 2>&1
    exit_code="$?"
    set -e
    if [[ "$exit_code" -eq 0 ]]; then
      printf 'completed\n' > "$run_dir/runner.status"
    else
      printf 'failed exit_code=%s\n' "$exit_code" > "$run_dir/runner.status"
    fi
    exit "$exit_code"
  ) &
  pid="$!"
  pids+=("$pid")
  printf '%s\n' "$pid" > "$run_dir/runner.pid"
  printf '%s=%s gpu=%s\n' "$label" "$pid" "$gpu" >> "$output_root/matrix.pids"
done

failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
if [[ "$failed" -eq 0 ]]; then
  printf 'completed\n' > "$output_root/matrix.status"
else
  printf 'failed\n' > "$output_root/matrix.status"
  exit 1
fi
