#!/usr/bin/env bash
set -euo pipefail

TRAIN_PATH=${TRAIN_PATH:-data/label/Standard/train.jsonl}
EVAL_PATH=${EVAL_PATH:-data/label/Standard/val.jsonl}
TEXT_FEAT_DIR=${TEXT_FEAT_DIR:-Soccer-GMR/feature/standard/clip_text}
CLIP_FEAT_DIR=${CLIP_FEAT_DIR:-Soccer-GMR/feature/standard/clip}
SLOWFAST_FEAT_DIR=${SLOWFAST_FEAT_DIR:-Soccer-GMR/feature/standard/slowfast}
RESULTS_DIR=${RESULTS_DIR:-results/moment_detr_gmr}
VARIANT=${VARIANT:-md_gmr}

python training/moment_detr_gmr/train.py \
  --dataset soccer_gmr \
  --feature clip_slowfast \
  --variant "${VARIANT}" \
  --train_path "${TRAIN_PATH}" \
  --eval_path "${EVAL_PATH}" \
  --t_feat_dir "${TEXT_FEAT_DIR}" \
  --v_feat_dirs "${CLIP_FEAT_DIR}" "${SLOWFAST_FEAT_DIR}" \
  --results_dir "${RESULTS_DIR}"
