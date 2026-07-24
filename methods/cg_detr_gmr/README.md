# CG-DETR / CG-DETR-GMR for Soccer-GMR

This directory is an isolated adaptation of the official CG-DETR code at
commit `212b2e49a0893512930d63b9294a8c066347e606`. The upstream MIT license is
preserved in `LICENSE`; `SOURCE.json` records the repository, pin, original
file hashes, and compatibility changes.

The model retains CG-DETR's dummy-token query representation, text-to-video
attention, distribution prompts, moment/sentence alignment, and conditional
DETR decoder. `cg_detr_gmr` adds the GMR paper's parallel existence branch:
max-pool the final decoder slots, then apply a two-layer ReLU MLP trained with
binary cross entropy. It does not alter localization logits or spans.

`cg_hiea2m` adds the unified HieA2M design. Because CG-DETR already contains
sentence and dummy-token ACA, it deliberately reuses that path and migrates
only DualGround's RPG/Slot/phrase-EOS temporal path, plus matched-IoU quality
calibration and factorized `{0,1,2,3,4+}` counting. The phrase and counter
residuals are exactly zero at parent warm-start, so localization and existence
outputs are bitwise identical to `cg_detr_gmr` at step zero.
The controlled `cg_quality`, `cg_phrase`, and `cg_counter` variants expose
each new branch independently for validation ablations. `cg_quality_phrase`
combines quality and phrase grounding while deliberately omitting the counter.

## MR-only protocol

- Inputs are normalized Soccer-GMR standard CLIP (512-D) + SlowFast (2304-D),
  two TEF coordinates, and token-level CLIP text (512-D).
- Annotation video IDs with a media extension resolve to an extension-free
  feature stem first and retain the extension as a fallback.
- `relevant_clips` is the binary union of all `relevant_windows`: clip and
  window half-open intervals must have positive overlap. A null query is an
  exact all-zero clip mask.
- CG's moment/sentence and distillation losses use only positive queries.
  Paper-literal GMR runs additionally pass `--mask-null-vmr-loss`: matcher
  classification/span/GIoU (main and auxiliary), quality, phrase grounding,
  and sample-dependent regularizers then exclude null rows, while the GMR
  existence loss remains active. The switch defaults to `false` only to load
  legacy checkpoints; in that compatibility mode DETR background
  classification still acts on null rows. The global prompt orthogonality term
  is a sample-independent parameter regularizer.
- The upstream QVHighlights saliency-ranking and negative-pair losses require
  per-clip QV ratings unavailable in Soccer-GMR. They are explicitly disabled;
  no pseudo-ratings are synthesized. `--no-use_cg_aux` provides a pure
  detection ablation.
- `cg_detr` uses positive-only training by default. `cg_detr_gmr` retains mixed
  positive/null training. Validation and test always retain every query and
  call `eval.evaluate_gmr`.

## Smoke regression

From the repository root:

```bash
python -m methods.cg_detr_gmr.smoke --device cuda
```

This runs finite forward/backward updates for positive, mixed, and fully-null
batches, evaluates the mixed predictions with the official evaluator, and
checks the existence-free baseline. Outputs are written under
`artifacts/smoke/cg_detr_gmr/` by default.

## Train and evaluate

```bash
python -m methods.cg_detr_gmr.train \
  --variant cg_detr \
  --output_dir artifacts/cg_detr_gmr/cg_detr

python -m methods.cg_detr_gmr.train \
  --variant cg_detr_gmr \
  --init_checkpoint artifacts/cg_detr_gmr/cg_detr/best_map.ckpt \
  --mask-null-vmr-loss \
  --lr 3e-5 \
  --output_dir artifacts/cg_detr_gmr/cg_detr_gmr

python -m methods.cg_detr_gmr.train \
  --variant cg_hiea2m \
  --init_checkpoint artifacts/cg_detr_gmr/cg_detr_gmr/best.ckpt \
  --mask-null-vmr-loss \
  --lr 3e-5 \
  --reference_map MATCHED_GMR_MAP \
  --reference_gmiou3 MATCHED_GMR_GMIOU3 \
  --output_dir artifacts/cg_detr_gmr/cg_hiea2m

python -m methods.cg_detr_gmr.evaluate \
  --checkpoint artifacts/cg_detr_gmr/cg_detr_gmr/best.ckpt \
  --eval_annotation data/label/Standard/test.jsonl
```

Training preserves separate best-mAP, best-G-mIoU@3, and best harmonic-joint
checkpoints plus their validation predictions and metrics. The existence-free
baseline selects by mAP; GMR variants select by the conservative minimum of
normalized mAP/G-mIoU@3 when matched references are supplied, and otherwise by
their harmonic mean.
For staged warm-starts, only the newly introduced existence/quality/phrase/
counter modules use the full requested learning rate; inherited parameters use
`--backbone_lr_scale` (default `0.1`). Exact grouping is saved to
`optimizer_groups.json`.
The default paths use `Soccer-GMR/feature/standard/{clip,slowfast,clip_text}`.
