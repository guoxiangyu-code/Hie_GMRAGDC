#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage:
  scripts/run_flash_vtg_strict.sh <variant> <physical-gpu-id> [seed] [batch-size] [run-dir]

Variants:
  plain, gmr, gmr_quality, gmr_zero, gmr_quality_zero

Defaults:
  seed       2023
  batch-size 8 (the released Flash-VTG/GMR setting)
  run-dir    artifacts/flash_vtg_supplement/seed<seed>/flash_vtg_<variant>

The physical GPU is isolated with CUDA_VISIBLE_DEVICES, so the training
program always receives --device 0.
EOF
}

if [[ $# -lt 2 || $# -gt 5 ]]; then
  usage
  exit 2
fi

variant="$1"
gpu_id="$2"
seed="${3:-2023}"
batch_size="${4:-8}"
eval_epoch="${FLASH_EVAL_EPOCH:-5}"
patience_epochs="${FLASH_PATIENCE:-80}"
cpu_threads="${GMR_CPU_THREADS:-6}"

case "$variant" in
  plain) run_name="flash_vtg_plain" ;;
  gmr) run_name="flash_vtg_gmr" ;;
  gmr_quality) run_name="flash_vtg_gmr_quality" ;;
  gmr_zero) run_name="flash_vtg_gmr_zero" ;;
  gmr_quality_zero) run_name="flash_vtg_gmr_quality_zero" ;;
  *)
    echo "unknown Flash-VTG variant: $variant" >&2
    usage
    exit 2
    ;;
esac

if [[ ! "$gpu_id" =~ ^[0-9]+$ ]]; then
  echo "physical-gpu-id must be a non-negative integer: $gpu_id" >&2
  exit 2
fi
if [[ ! "$seed" =~ ^[0-9]+$ ]]; then
  echo "seed must be a non-negative integer: $seed" >&2
  exit 2
fi
if [[ ! "$batch_size" =~ ^[1-9][0-9]*$ ]]; then
  echo "batch-size must be a positive integer: $batch_size" >&2
  exit 2
fi
if [[ ! "$eval_epoch" =~ ^[1-9][0-9]*$ ]]; then
  echo "FLASH_EVAL_EPOCH must be a positive integer: $eval_epoch" >&2
  exit 2
fi
if [[ ! "$patience_epochs" =~ ^[1-9][0-9]*$ ]]; then
  echo "FLASH_PATIENCE must be a positive integer: $patience_epochs" >&2
  exit 2
fi
if [[ ! "$cpu_threads" =~ ^[4-8]$ ]]; then
  echo "GMR_CPU_THREADS must be between 4 and 8: $cpu_threads" >&2
  exit 2
fi
patience_evals=$(((patience_epochs + eval_epoch - 1) / eval_epoch))

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python_bin="/home/guoxiangyu/miniconda3/bin/python"
train_path="$repo_root/data/label/Standard/train.jsonl"
val_path="$repo_root/data/label/Standard/val.jsonl"
slowfast_dir="$repo_root/Soccer-GMR/feature/standard/slowfast"
clip_dir="$repo_root/Soccer-GMR/feature/standard/clip"
text_dir="$repo_root/Soccer-GMR/feature/standard/clip_text"
run_dir="${5:-$repo_root/artifacts/flash_vtg_supplement/seed${seed}/${run_name}}"

if [[ "$run_dir" != /* ]]; then
  run_dir="$repo_root/$run_dir"
fi
run_dir="$(realpath -m "$run_dir")"
case "$run_dir" in
  "$repo_root"/*) ;;
  *)
    echo "run-dir must be inside $repo_root: $run_dir" >&2
    exit 2
    ;;
esac

required_paths=(
  "$python_bin"
  "$train_path"
  "$val_path"
  "$slowfast_dir"
  "$clip_dir"
  "$text_dir"
  "$repo_root/configs/flash_vtg_gmr/model.py"
)
for path in "${required_paths[@]}"; do
  if [[ ! -e "$path" ]]; then
    echo "missing required path: $path" >&2
    exit 3
  fi
done

mkdir -p "$run_dir"
pid_file="$run_dir/runner.pid"
if [[ -s "$pid_file" ]]; then
  old_pid="$(<"$pid_file")"
  if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "refusing to launch over live PID $old_pid in $run_dir" >&2
    exit 4
  fi
fi
if [[ "${FLASH_ALLOW_EXISTING:-0}" != "1" ]] \
  && { [[ -f "$run_dir/opt.json" ]] || [[ -f "$run_dir/model_latest.ckpt" ]]; }; then
  echo "run-dir already contains a Flash training run: $run_dir" >&2
  echo "set FLASH_ALLOW_EXISTING=1 only if intentional" >&2
  exit 4
fi

common_args=(
  "$repo_root/configs/flash_vtg_gmr/model.py"
  --dset_name hl
  --ctx_mode video_tef
  --train_path "$train_path"
  --eval_path "$val_path"
  --eval_split_name val
  --v_feat_dirs "$slowfast_dir" "$clip_dir"
  --t_feat_dir "$text_dir"
  --v_feat_dim 2816
  --t_feat_dim 512
  --max_q_l 40
  --max_v_l 75
  --clip_length 2
  --max_windows 5
  --lr "${FLASH_LR:-3e-5}"
  --lr_drop "${FLASH_LR_DROP:-400}"
  --wd "${FLASH_WD:-1e-4}"
  --n_epoch "${FLASH_EPOCHS:-400}"
  --max_es_cnt "$patience_evals"
  --bsz "$batch_size"
  --eval_bsz 1
  --eval_epoch "$eval_epoch"
  --selection_metric "${FLASH_SELECTION_METRIC:-mAP}"
  --reference_map "${FLASH_REFERENCE_MAP:-0.0}"
  --reference_gmiou3 "${FLASH_REFERENCE_GMIOU3:-0.0}"
  --data_ratio "${FLASH_DATA_RATIO:-1.0}"
  --num_workers "${FLASH_NUM_WORKERS:-0}"
  --device 0
  --run_dir "$run_dir"
  --exp_id "$run_name"
  --seed "$seed"
  --hidden_dim 256
  --dim_feedforward 1024
  --enc_layers 3
  --t2v_layers 6
  --dummy_layers 2
  --nheads 8
  --num_dummies 40
  --total_prompts 10
  --num_prompts 1
  --kernel_size 5
  --num_conv_layers 1
  --num_mlp_layers 5
  --use_SRM
  --input_dropout 0.5
  --dropout 0.1
  --span_loss_type l1
  --lw_reg 1.0
  --lw_cls 5.0
  --lw_sal 0.0
  --lw_saliency 0.0
  --lw_wattn 1.0
  --lw_ms_align 1.0
  --mr_only
  --eval_full_only
  --gmr_cls_thresholds 0.4 0.6 0.8
  --gmiou_cls_threshold 0.4
  --nms_thd 0.7
)

if [[ "$variant" != "plain" ]]; then
  common_args+=(
    --use_exist_head
    --exist_pool mean
    --exist_loss_coef 1.0
    --exist_gate_thd 0.5
  )
fi

case "$variant" in
  gmr_quality|gmr_quality_zero)
    common_args+=(
      --use_quality_head
      --quality_loss_coef "${FLASH_QUALITY_LOSS_COEF:-1.0}"
      --quality_score_alpha "${FLASH_QUALITY_SCORE_ALPHA:-0.5}"
      --quality_negative_weight "${FLASH_QUALITY_NEGATIVE_WEIGHT:-0.1}"
    )
    ;;
esac

case "$variant" in
  gmr_zero|gmr_quality_zero)
    common_args+=(
      --use_independent_zero_head
      --zero_loss_coef "${FLASH_ZERO_LOSS_COEF:-1.0}"
      --zero_positive_query_weight "${FLASH_ZERO_POSITIVE_QUERY_WEIGHT:-1.0}"
      --exist_loose_thd "${FLASH_EXIST_LOOSE_THD:-0.35}"
      --zero_rescue_thd "${FLASH_ZERO_RESCUE_THD:-0.45}"
      --zero_veto_thd "${FLASH_ZERO_VETO_THD:-0.65}"
      --zero_weak_candidate_thd "${FLASH_ZERO_WEAK_CANDIDATE_THD:-0.35}"
    )
    ;;
esac

if [[ "$variant" == gmr_quality* || "$variant" == "gmr_zero" ]]; then
  init_checkpoint="${FLASH_INIT_CHECKPOINT:-$repo_root/Soccer-GMR/checkpoint/flashVTG_gmr}"
  if [[ ! -f "$init_checkpoint" ]]; then
    echo "missing Flash parent checkpoint: $init_checkpoint" >&2
    exit 3
  fi
  common_args+=(
    --resume "$init_checkpoint"
    --allow_head_init
    --freeze_parent
  )
  if [[ "$variant" == "gmr_quality_zero" && "${FLASH_FREEZE_QUALITY:-0}" == "1" ]]; then
    common_args+=(--freeze_quality_head)
  fi
fi

if [[ -n "${FLASH_RESUME_CHECKPOINT:-}" ]]; then
  if [[ ! -f "$FLASH_RESUME_CHECKPOINT" ]]; then
    echo "missing Flash exact-resume checkpoint: $FLASH_RESUME_CHECKPOINT" >&2
    exit 3
  fi
  common_args+=(--resume "$FLASH_RESUME_CHECKPOINT" --resume_all)
fi

if [[ "${FLASH_DEBUG:-0}" == "1" ]]; then
  common_args+=(--debug)
fi

{
  printf 'variant=%s\n' "$variant"
  printf 'physical_gpu=%s\n' "$gpu_id"
  printf 'logical_device=0\n'
  printf 'python=%s\n' "$python_bin"
  printf 'seed=%s\n' "$seed"
  printf 'batch_size=%s\n' "$batch_size"
  printf 'eval_every_epochs=%s\n' "$eval_epoch"
  printf 'patience_epochs=%s\npatience_evaluations=%s\n' \
    "$patience_epochs" "$patience_evals"
  printf 'cpu_threads=%s\n' "$cpu_threads"
  printf 'started_at=%s\n' "$(date --iso-8601=seconds)"
  printf 'git_commit=%s\n' "$(git rev-parse HEAD 2>/dev/null || printf unknown)"
} > "$run_dir/launch_metadata.txt"
{
  printf 'CUDA_VISIBLE_DEVICES=%q ' "$gpu_id"
  printf '%q ' "$python_bin" -u -m training.flash_vtg_gmr.train "${common_args[@]}"
  printf '\n'
} > "$run_dir/launch_command.txt"

printf '%s\n' "$$" > "$pid_file"
printf 'running\n' > "$run_dir/runner.status"
finish() {
  local exit_code="$?"
  if [[ "$exit_code" -eq 0 ]]; then
    printf 'completed\n' > "$run_dir/runner.status"
  else
    printf 'failed exit_code=%s\n' "$exit_code" > "$run_dir/runner.status"
  fi
}
trap finish EXIT

env \
  CUDA_VISIBLE_DEVICES="$gpu_id" \
  PYTHONUNBUFFERED=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  OMP_NUM_THREADS="$cpu_threads" \
  MKL_NUM_THREADS="$cpu_threads" \
  OPENBLAS_NUM_THREADS="$cpu_threads" \
  NUMEXPR_NUM_THREADS="$cpu_threads" \
  "$python_bin" -u -m training.flash_vtg_gmr.train \
  "${common_args[@]}" 2>&1 | tee -a "$run_dir/stdout.log"
