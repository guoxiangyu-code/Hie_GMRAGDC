# EaTR / EaTR-GMR for Soccer-GMR

This directory is an isolated, MIT-licensed adaptation of the official EaTR
implementation pinned in `UPSTREAM.json`. It does not import or modify the
repository's Moment-DETR implementation.

Six controlled variants are available:

- `eatr`: the original event-aware EaTR localization backbone. For GMR
  evaluation, the official evaluator uses maximum window confidence as its
  fallback existence score.
- `eatr_gmr`: the same backbone plus the Soccer-GMR paper's parallel existence
  adapter: max-pool final moment decoder queries, then a two-layer ReLU MLP and
  BCE-with-logits supervision (`lambda_exist=1` by default).
- `eatr_quality`: `eatr_gmr` plus a final-query temporal-IoU quality head.
- `eatr_dual`: `eatr_gmr` plus sentence/phrase DualGround interaction.
- `eatr_counter`: `eatr_gmr` plus hierarchical `{0,1,2,3,4+}` counting.
- `eatr_hiea2m`: quality, DualGround, and hierarchical counting together.

All variants use MR-only losses: foreground/background query classification,
span L1/GIoU, and EaTR pseudo-event L1/GIoU. Saliency and QVHighlights-only
labels are not required. Positive, mixed, and all-null batches are supported.
For a paper-literal mixed GMR run, `--mask-null-vmr-loss` applies the GMR
indicator to moment classification/span/GIoU, auxiliary layers, pseudo-event,
quality, and DualGround losses. Null rows then supervise only existence;
positive-conditional counter terms remain positive-only. The flag defaults to
`false` solely for legacy-checkpoint compatibility and is part of exact-resume
configuration validation.

DualGround is injected after EaTR's video/text projections and before their
concatenation, so its refined video enters both event and moment reasoning. The
quality, existence, and counter heads consume only the final moment query
states `hs[-1]`; the event query `hs[0]` is never used by those heads. The
counter additionally pools masked encoder video/text memory.

Optional branches preserve an `eatr_gmr` parent at step zero: DualGround masks
only its zero-gated residual, the counter existence head is a zero-initialized
residual over the parent existence logit, and the quality head's last layer is
zero initialized. Thus parent logits, spans, existence scores, and query
ranking are unchanged immediately after warm-start.

## Feature layout

Extract the Soccer-GMR standard archives into three directories:

```text
slowfast_dir/<video-stem>.npz       key: features, 2304 channels
clip_dir/<video-stem>.npz           key: features, 512 channels
text_dir/qid<qid>.npz               key: last_hidden_state, 512 channels
```

SlowFast and CLIP are row-normalized, concatenated in that order, and augmented
with two temporal-endpoint features. Annotation ids ending in `.mp4` (and other
common video extensions) are mapped to extension-free NPZ stems.

## Training

```bash
python -m methods.eatr_gmr.train \
  --variant eatr_hiea2m \
  --init-checkpoint artifacts/eatr_gmr/best.pt \
  --mask-null-vmr-loss \
  --train-annotations data/label/Standard/train.jsonl \
  --val-annotations data/label/Standard/val.jsonl \
  --slowfast-dir /path/to/slowfast \
  --clip-dir /path/to/clip \
  --text-dir /path/to/clip_text \
  --output-dir artifacts/eatr_gmr
```

Omit `--init-checkpoint` to train from scratch. `--resume` instead restores an
exact same-structure run including optimizer/scheduler state. Defaults reproduce
the upstream QV model shape (`hidden_dim=256`, three encoder/decoder layers, ten
queries) and the GMR paper learning rate of `3e-5`. Optional branch parameters
use the requested learning rate while the warm-started backbone uses
`--backbone-lr-scale 0.1`. The same staged rule applies when creating
`eatr_gmr`: the new existence head uses the full rate and inherited EaTR
parameters use the scaled rate. `run.json` records both groups and their exact
parameter counts.

Validation and checkpoint selection always use the full ranked query set. For
counter variants, a separate `*_adaptive.jsonl` and metric file is also saved;
this diagnostic never replaces the primary full-set result.

## Evaluation

```bash
python -m methods.eatr_gmr.evaluate \
  --checkpoint artifacts/eatr_gmr/best.pt \
  --annotations data/label/Standard/val.jsonl \
  --slowfast-dir /path/to/slowfast \
  --clip-dir /path/to/clip \
  --text-dir /path/to/clip_text \
  --output-dir artifacts/eatr_gmr_eval
```

Evaluation calls `eval.eval_main.evaluate_gmr` directly. Use `--no-metrics` for
unlabeled test prediction. Checkpoint structure is detected from state-dict
keys rather than trusting possibly stale config flags. The primary files are
always `submission.jsonl` and `metrics.json` with full decoding. Counter
checkpoints additionally produce `submission_adaptive.jsonl` and
`metrics_adaptive.json`; pass `--no-adaptive-diagnostics` to disable them.

Both views include factorized count probabilities and the decoded count, so the
official evaluator reports `Count-Acc`, `Count-MacroAcc`,
`Positive-Count-Acc`, NLL, and Brier score when available. Adaptive decoding is
explicitly two-stage: quality/diversity query ranking first, hierarchical
existence and positive-conditional cardinality selection second.

## Smoke verification

```bash
python -m methods.eatr_gmr.smoke
python -m unittest -v tests.test_eatr_gmr_hiea2m
```

The smoke suite creates temporary NPZ features and exercises all six variants
on positive-only, mixed, and all-null batches, including backward passes,
conditional-count zero losses for all-null batches, full/adaptive decoding,
Count metrics, and the official evaluator. The unittest additionally verifies
bitwise step-zero parent preservation and automatic checkpoint detection.
