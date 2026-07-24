# QD-DETR / QD-DETR-GMR for Soccer-GMR

This directory is an isolated adaptation of the official QD-DETR code at
commit `f8628f79f7c651b586300b142dbe9b85e43857cc`. The upstream MIT license is
preserved in `LICENSE`; `SOURCE.json` records provenance, original file hashes,
and every compatibility change made here.

The model keeps QD-DETR's query-dependent T2V encoder and conditional DETR
decoder. The `qd_detr_gmr` variant adds the GMR paper's parallel existence
branch: max-pool the final decoder slots, apply a two-layer ReLU MLP, and train
with binary cross entropy. Localization logits/spans remain unchanged.

The unified HieA2M variants reuse the same shared implementations as the
Moment-DETR path:

- `qd_quality`: matched-IoU query quality calibration;
- `qd_dual`: DualGround sentence/phrase conditioning after QD's input
  projections and before video/text concatenation;
- `qd_quality_dual`: quality + DualGround, without the hierarchical counter;
- `qd_counter`: factorized existence plus positive-conditional `{1,2,3,4+}`
  counting;
- `qd_hiea2m`: quality + DualGround + hierarchical counter.

All four retain the paper-style existence adapter. DualGround's temporal
residual is exactly zero at warm-start, the counter contributes an exactly
zero existence residual, and constant initial quality preserves query order.
QD's rolled negative saliency pass remains disabled under the MR-only protocol;
it is never synthesized using positive-conditioned video features.

## Protocol

- Input: normalized Soccer-GMR standard CLIP (512-D) + SlowFast (2304-D),
  two TEF coordinates, and token-level CLIP text (512-D).
- The parent QD path can retain its fixed 32-token layout for checkpoint
  compatibility, while DualGround/counter receive a separate mask from the
  stored CLIP `attention_mask`, so sentence EOS and lexical pooling never use
  padding as language evidence.
- Video IDs ending in `.mp4` are resolved using their extension-free feature
  stem, with a retained-extension fallback.
- `qd_detr` defaults to positive-only training, matching a conventional MR
  baseline. It has no explicit existence score; the shared evaluator falls
  back to the maximum foreground query score.
- `qd_detr_gmr` defaults to mixed positive/null training and emits
  `pred_exist_score`. Fully null batches are valid.
- Paper-literal GMR training must pass `--mask-null-vmr-loss`. This applies the
  GMR indicator to matcher-based classification/span/GIoU losses (including
  auxiliary decoder layers) and to sample-dependent quality/DualGround losses,
  so a null query supervises the existence decision only. The switch defaults
  to `false` solely for old-checkpoint compatibility; full resume rejects a
  change in this loss semantic.
- This is MR-only. QD-DETR's highlight/saliency branch is disabled; no
  pseudo-highlight labels are synthesized from moment annotations.
- Validation/test always retain all queries and use `eval.evaluate_gmr`.

## Smoke tests

From the repository root:

```bash
python -m methods.qd_detr_gmr.smoke --device cuda
```

This executes a finite `qd_hiea2m` forward/backward step for positive, mixed,
and all-null batches, serializes each prediction set, runs the official
evaluator (including count diagnostics) on the mixed subset, and checks the
baseline without an existence head. Its second stage migrates a trained parent
adapter into HieA2M and asserts exact localization/existence warm-start,
positive count-head gradients, zero conditional-count loss/gradient on an
all-null batch, and nonzero existence gradient. Outputs go to
`artifacts/smoke/qd_detr_gmr/` by default.

## Training

Official QD-DETR baseline defaults (2 encoder/2 decoder layers, 256 hidden,
10 queries, AdamW `1e-4`) are exposed by the CLI:

```bash
python -m methods.qd_detr_gmr.train \
  --variant qd_detr \
  --output_dir artifacts/qd_detr_gmr/qd_detr
```

Train the existence-adapted model from the matched baseline (the missing
existence keys are written to `initialization_audit.json`):

```bash
python -m methods.qd_detr_gmr.train \
  --variant qd_detr_gmr \
  --init_checkpoint artifacts/qd_detr_gmr/qd_detr/best_map.ckpt \
  --mask-null-vmr-loss \
  --lr 3e-5 \
  --output_dir artifacts/qd_detr_gmr/qd_detr_gmr
```

In this stage the new `exist_head` receives `--lr` and every inherited QD
parameter receives `--lr * --backbone_lr_scale`. The exact prefixes, learning
rates, and parameter counts are frozen in `optimizer_groups.json`.

For a strictly matched staged run, initialize `qd_quality`, `qd_dual`,
`qd_quality_dual`, `qd_counter`, or `qd_hiea2m` from the same `qd_detr_gmr`
checkpoint. Newly introduced parameters use the requested learning rate while
shared QD/GMR parameters default to `0.1x`; control this with
`--backbone_lr_scale`.

```bash
python -m methods.qd_detr_gmr.train \
  --variant qd_hiea2m \
  --init_checkpoint artifacts/qd_detr_gmr/qd_detr_gmr/best.ckpt \
  --mask-null-vmr-loss \
  --lr 3e-5 \
  --output_dir artifacts/qd_detr_gmr/qd_hiea2m
```

Each run preserves `best_map.ckpt`, `best_g_miou3.ckpt`, `best_joint.ckpt`,
`best.ckpt` (mAP for baseline, joint mAP/G-mIoU@3 for GMR), latest outputs,
and JSONL training logs. With `--reference_map` and `--reference_gmiou3`, joint
selection is the conservative minimum of the two normalized ratios; otherwise
it is their harmonic mean. Use `--train_sample_mode positive|mixed|null` for an
explicit ablation; `auto` is the protocol default above.

The primary validation/test submission is always `full`: every DETR slot is
retained after quality/diversity ranking, so a young count head cannot destroy
localization recall. For HieA2M variants the same forward pass also saves a
GREC-style `threshold` set and HieA2G-style `adaptive` count set as diagnostics;
neither controls checkpoint selection. Disable these extra metric passes with
`--no-diagnostic_decoders`.

## Test evaluation

```bash
python -m methods.qd_detr_gmr.evaluate \
  --checkpoint artifacts/qd_detr_gmr/qd_detr_gmr/best.ckpt \
  --eval_annotation data/label/Standard/test.jsonl
```

The default feature paths point at
`Soccer-GMR/feature/standard/{clip,slowfast,clip_text}`. All paths, batch sizes,
the existence gate used for G-mIoU, and rounded versus continuous timestamps
are CLI-configurable; run each module with `--help` for the full contract.
Evaluation detects existence/quality/DualGround/counter structure directly
from checkpoint keys and reports any saved-config mismatch before strict load.
