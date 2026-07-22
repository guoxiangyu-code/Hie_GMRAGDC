#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH=${MODEL_PATH:-Soccer-GMR/checkpoint/moment_detr_gmr}
SPLIT=${SPLIT:-test}
EVAL_PATH=${EVAL_PATH:-data/label/Standard/test.jsonl}
TEXT_FEAT_DIR=${TEXT_FEAT_DIR:-Soccer-GMR/feature/standard/clip_text}
CLIP_FEAT_DIR=${CLIP_FEAT_DIR:-Soccer-GMR/feature/standard/clip}
SLOWFAST_FEAT_DIR=${SLOWFAST_FEAT_DIR:-Soccer-GMR/feature/standard/slowfast}
RESULTS_DIR=${RESULTS_DIR:-results/moment_detr_gmr/${SPLIT}}

python training/moment_detr_gmr/evaluate.py \
  --dataset soccer_gmr \
  --feature clip_slowfast \
  --model_path "${MODEL_PATH}" \
  --split "${SPLIT}" \
  --eval_path "${EVAL_PATH}" \
  --t_feat_dir "${TEXT_FEAT_DIR}" \
  --v_feat_dirs "${CLIP_FEAT_DIR}" "${SLOWFAST_FEAT_DIR}" \
  --results_dir "${RESULTS_DIR}"
