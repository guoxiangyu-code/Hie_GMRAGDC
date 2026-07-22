"""Self-contained positive/mixed/all-null smoke suite."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .build import build_model
from .config import EaTRConfig
from .dataset import SoccerGMRDataset, collate_fn, move_batch
from .runtime import official_metrics, predict_views, seed_everything
from .variants import VARIANT_FLAGS, apply_variant


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def _make_fixture(root: Path):
    slowfast_dir = root / "slowfast"
    clip_dir = root / "clip"
    text_dir = root / "clip_text"
    for directory in (slowfast_dir, clip_dir, text_dir):
        directory.mkdir()

    cases = {
        "positive": [
            {"qid": 1, "query": "first action", "vid": "positive_a.mp4",
             "duration": 16.0, "relevant_windows": [[2.0, 6.0]]},
            {"qid": 2, "query": "two actions", "vid": "positive_b.mp4",
             "duration": 16.0, "relevant_windows": [[0.0, 4.0], [8.0, 12.0]]},
        ],
        "mixed": [
            {"qid": 3, "query": "visible event", "vid": "mixed_a.mp4",
             "duration": 16.0, "relevant_windows": [[4.0, 10.0]]},
            {"qid": 4, "query": "absent event", "vid": "mixed_b.mp4",
             "duration": 16.0, "relevant_windows": []},
        ],
        "all_null": [
            {"qid": 5, "query": "nothing one", "vid": "null_a.mp4",
             "duration": 16.0, "relevant_windows": []},
            {"qid": 6, "query": "nothing two", "vid": "null_b.mp4",
             "duration": 16.0, "relevant_windows": []},
        ],
    }

    rng = np.random.default_rng(7)
    for rows in cases.values():
        for row in rows:
            # Annotation ids retain .mp4 while feature names deliberately do not.
            stem = Path(row["vid"]).stem
            np.savez(
                slowfast_dir / f"{stem}.npz",
                features=rng.normal(size=(8, 6)).astype(np.float32),
            )
            np.savez(
                clip_dir / f"{stem}.npz",
                features=rng.normal(size=(8, 4)).astype(np.float32),
            )
            np.savez(
                text_dir / f"qid{row['qid']}.npz",
                last_hidden_state=rng.normal(size=(6, 8)).astype(np.float32),
                attention_mask=np.array([1, 1, 1, 1, 0, 0], dtype=np.int64),
            )

    annotations = {}
    for name, rows in cases.items():
        path = root / f"{name}.jsonl"
        _write_jsonl(path, rows)
        annotations[name] = path
    return annotations, slowfast_dir, clip_dir, text_dir


def _config(variant: str) -> EaTRConfig:
    config = EaTRConfig(
        video_dim=12,
        text_dim=8,
        hidden_dim=16,
        nheads=4,
        enc_layers=1,
        dec_layers=1,
        dim_feedforward=32,
        dropout=0.0,
        input_dropout=0.0,
        num_queries=4,
        num_slot_iter=1,
        n_input_proj=1,
        max_q_l=6,
        max_v_l=8,
        aux_loss=False,
        counter_dropout=0.0,
    )
    return apply_variant(config, variant)


def run_smoke() -> dict:
    seed_everything(11)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    device = torch.device("cpu")
    report = {"status": "ok", "variants": {}}
    try:
        with tempfile.TemporaryDirectory(prefix="eatr_gmr_smoke_") as temp_dir:
            root = Path(temp_dir)
            annotations, slowfast_dir, clip_dir, text_dir = _make_fixture(root)
            for variant in VARIANT_FLAGS:
                config = _config(variant)
                model, criterion = build_model(config)
                model.to(device)
                criterion.to(device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
                variant_report = {"batches": {}}
                datasets = {}

                for case_name, annotation_path in annotations.items():
                    dataset = SoccerGMRDataset(
                        annotation_path,
                        slowfast_dir,
                        clip_dir,
                        text_dir,
                        max_video_len=8,
                        max_text_len=6,
                        clip_length=2.0,
                        max_windows=4,
                        load_labels=True,
                        expected_video_feature_dim=10,
                        expected_text_feature_dim=8,
                    )
                    datasets[case_name] = dataset
                    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
                    _, inputs, targets = next(iter(loader))
                    inputs, targets = move_batch(inputs, targets, device)
                    outputs = model(**inputs)
                    losses = criterion(outputs, targets)
                    total = criterion.weighted_loss(losses)
                    if not torch.isfinite(total):
                        raise AssertionError(f"{variant}/{case_name}: non-finite total loss")
                    if case_name == "all_null":
                        if float(losses["loss_span"].detach()) != 0.0:
                            raise AssertionError("all-null span loss must be exactly zero")
                        if float(losses["loss_giou"].detach()) != 0.0:
                            raise AssertionError("all-null GIoU loss must be exactly zero")
                    if config.use_exist_head != ("pred_exist_logits" in outputs):
                        raise AssertionError("existence head output does not match variant")
                    if case_name == "all_null" and config.use_hierarchical_counter:
                        for name in (
                            "loss_count", "loss_count_ordinal",
                            "loss_count_contrastive", "loss_count_consistency",
                        ):
                            if float(losses[name].detach()) != 0.0:
                                raise AssertionError(
                                    f"all-null conditional loss must be zero: {name}"
                                )

                    optimizer.zero_grad(set_to_none=True)
                    total.backward()
                    finite_gradients = [
                        torch.isfinite(parameter.grad).all().item()
                        for parameter in model.parameters() if parameter.grad is not None
                    ]
                    if not finite_gradients or not all(finite_gradients):
                        raise AssertionError(f"{variant}/{case_name}: invalid gradients")
                    optimizer.step()
                    variant_report["batches"][case_name] = {
                        "total_loss": round(float(total.detach()), 6),
                        "num_positive": int(targets["exist_label"].sum().item()),
                        "span_shape": [
                            list(target["spans"].shape) for target in targets["span_labels"]
                        ],
                    }

                mixed_loader = DataLoader(
                    datasets["mixed"], batch_size=2, collate_fn=collate_fn
                )
                modes = (
                    ("full", "adaptive")
                    if config.use_hierarchical_counter else ("full",)
                )
                views = predict_views(
                    model, mixed_loader, device, modes=modes, max_predictions=4,
                    round_to_clip=True, clip_length=2.0,
                )
                submission = views["full"]
                metrics = official_metrics(
                    submission, datasets["mixed"].data, map_num_workers=1
                )
                variant_report["official_evaluator"] = {
                    "num_predictions": len(submission),
                    "has_explicit_exist_score": "pred_exist_score" in submission[0],
                    "has_quality": config.use_quality_head,
                    "has_count_metrics": "Count" in metrics,
                    "brief_keys": sorted(metrics["brief"].keys()),
                }
                if config.use_hierarchical_counter:
                    variant_report["adaptive_diagnostic"] = {
                        "num_predictions": len(views["adaptive"]),
                        "window_counts": [
                            len(row["pred_relevant_windows"])
                            for row in views["adaptive"]
                        ],
                    }
                report["variants"][variant] = variant_report
    finally:
        torch.set_num_threads(previous_threads)
    return report


def main() -> None:
    print(json.dumps(run_smoke(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
