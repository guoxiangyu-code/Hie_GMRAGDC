#!/usr/bin/env python3
"""Measure a real Moment-DETR training step at one candidate batch size.

Each invocation is intentionally a fresh process so a failed CUDA allocation
cannot contaminate the next candidate.  The probe runs two full
forward/backward/AdamW steps on a maximal-length positive Soccer-GMR example;
the first step materializes optimizer state and both steps contribute to the
reported CUDA peak.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.moment_detr_gmr.hierarchical_counter import (  # noqa: E402
    inverse_sqrt_positive_count_weights,
)
from training.moment_detr_gmr.config import BaseOptions  # noqa: E402
from training.moment_detr_gmr.dataset import (  # noqa: E402
    StartEndDataset,
    prepare_batch_inputs,
    start_end_collate,
)
from training.moment_detr_gmr.evaluate import setup_model  # noqa: E402
from training.moment_detr_gmr.train import build_dataset_config, set_seed  # noqa: E402


VARIANTS = {
    "md_gmr": (True, False, False, False),
    "md_hiea2m": (True, True, True, True),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe a full Moment-DETR optimizer step at one batch size.",
        allow_abbrev=False,
    )
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="md_hiea2m")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--result-json", type=Path)
    parser.add_argument("--train-path", type=Path, default=REPO_ROOT / "data/label/Standard/train.jsonl")
    parser.add_argument(
        "--text-feature-dir",
        type=Path,
        default=REPO_ROOT / "Soccer-GMR/feature/standard/clip_text",
    )
    parser.add_argument(
        "--video-feature-dirs",
        type=Path,
        nargs=2,
        default=[
            REPO_ROOT / "Soccer-GMR/feature/standard/clip",
            REPO_ROOT / "Soccer-GMR/feature/standard/slowfast",
        ],
    )
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.steps < 2:
        parser.error("--steps must be at least 2 so AdamW state is resident")
    return args


def build_options(args: argparse.Namespace):
    manager = BaseOptions("moment_detr", "soccer_gmr", "clip_slowfast", resume=None)
    manager.parse()
    opt = manager.option
    opt.seed = args.seed
    opt.device = "cuda"
    opt.bsz = args.batch_size
    opt.num_workers = 0
    opt.train_path = str(args.train_path.resolve())
    opt.t_feat_dir = str(args.text_feature_dir.resolve())
    opt.v_feat_dirs = [str(path.resolve()) for path in args.video_feature_dirs]
    opt.mr_only = True
    opt.lw_saliency = 0
    opt.variant = args.variant
    (
        opt.use_exist_head,
        opt.use_quality_head,
        opt.use_dual_grounding,
        opt.use_hierarchical_counter,
    ) = VARIANTS[args.variant]
    opt.mask_null_vmr_loss = True
    # Match the released Soccer-GMR input and output protocol.
    opt.trim_text_by_attention_mask = False
    opt.round_to_clip = True
    return opt


def build_probe_sample(dataset: StartEndDataset, opt):
    positive_indices = [
        index
        for index, row in enumerate(dataset.data)
        if len(row.get("relevant_windows") or []) > 0
    ]
    if not positive_indices:
        raise RuntimeError("training split contains no positive sample")

    # More targets increase matching work.  Among those rows, retain the
    # longest actual video/text tensors and stop at both configured caps.
    positive_indices.sort(
        key=lambda index: len(dataset.data[index].get("relevant_windows") or []),
        reverse=True,
    )
    best = None
    best_key = None
    max_targets = len(dataset.data[positive_indices[0]].get("relevant_windows") or [])
    for index in positive_indices:
        target_count = len(dataset.data[index].get("relevant_windows") or [])
        if target_count < max_targets and best is not None:
            break
        item = dataset[index]
        query_length = int(item["model_inputs"]["query_feat"].shape[0])
        video_length = int(item["model_inputs"]["video_feat"].shape[0])
        key = (video_length + query_length, video_length, query_length, target_count)
        if best_key is None or key > best_key:
            best = item
            best_key = key
        if video_length >= int(opt.max_v_l) and query_length >= int(opt.max_q_l):
            break
    if best is None:
        raise RuntimeError("failed to construct a positive probe sample")
    return best


def write_result(result: dict, destination: Path | None) -> None:
    encoded = json.dumps(result, sort_keys=True)
    print(f"PROBE_RESULT={encoded}", flush=True)
    if destination is not None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(encoded + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    started = time.monotonic()
    result = {
        "schema_version": 1,
        "variant": args.variant,
        "batch_size": args.batch_size,
        "steps": args.steps,
        "status": "failed",
        "mask_null_vmr_loss": True,
        "trim_text_by_attention_mask": False,
        "round_to_clip": True,
    }

    try:
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError(
                "probe requires exactly one CUDA device after CUDA_VISIBLE_DEVICES filtering"
            )
        torch.cuda.set_device(0)
        set_seed(args.seed, use_cuda=True)
        opt = build_options(args)
        dataset = StartEndDataset(
            **build_dataset_config(opt, opt.train_path, load_labels=True, keep_empty_gt=True)
        )
        if len(dataset) != sum(1 for _ in open(opt.train_path, "r", encoding="utf-8")):
            raise RuntimeError("incomplete training feature coverage")
        if opt.use_hierarchical_counter:
            counts = [0, 0, 0, 0]
            for row in dataset.data:
                count = min(len(row.get("relevant_windows") or []), 4)
                if count > 0:
                    counts[count - 1] += 1
            opt.positive_count_weights = inverse_sqrt_positive_count_weights(counts).tolist()

        sample = build_probe_sample(dataset, opt)
        sample_inputs = sample["model_inputs"]
        result["sample_qid"] = sample["meta"].get("qid")
        result["query_length"] = int(sample_inputs["query_feat"].shape[0])
        result["video_length"] = int(sample_inputs["video_feat"].shape[0])
        result["target_count"] = int(sample_inputs["span_labels"].shape[0])

        batch = start_end_collate([sample] * args.batch_size)
        model, criterion, optimizer, _ = setup_model(opt)
        model.train()
        criterion.train()
        model_inputs, targets = prepare_batch_inputs(batch[1], "cuda")
        torch.cuda.synchronize()
        free_before, total_bytes = torch.cuda.mem_get_info()
        torch.cuda.reset_peak_memory_stats()

        last_loss = None
        for _ in range(args.steps):
            optimizer.zero_grad(set_to_none=True)
            outputs = model(**model_inputs)
            loss_dict = criterion(outputs, targets)
            losses = sum(
                loss_dict[name] * criterion.weight_dict[name]
                for name in loss_dict
                if name in criterion.weight_dict
            )
            if not bool(torch.isfinite(losses).item()):
                raise RuntimeError(f"non-finite probe loss: {float(losses.detach())}")
            losses.backward()
            if opt.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
            optimizer.step()
            last_loss = float(losses.detach())
            del outputs, loss_dict, losses
        torch.cuda.synchronize()

        result.update(
            status="ok",
            loss=last_loss,
            device_name=torch.cuda.get_device_name(0),
            total_mib=round(total_bytes / 2**20, 2),
            free_before_step_mib=round(free_before / 2**20, 2),
            peak_allocated_mib=round(torch.cuda.max_memory_allocated() / 2**20, 2),
            peak_reserved_mib=round(torch.cuda.max_memory_reserved() / 2**20, 2),
            end_allocated_mib=round(torch.cuda.memory_allocated() / 2**20, 2),
            end_reserved_mib=round(torch.cuda.memory_reserved() / 2**20, 2),
        )
    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        is_oom = isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()
        result["status"] = "oom" if is_oom else "failed"
        result["error"] = str(exc)
        if torch.cuda.is_available():
            try:
                result["peak_allocated_mib"] = round(
                    torch.cuda.max_memory_allocated() / 2**20, 2
                )
                result["peak_reserved_mib"] = round(
                    torch.cuda.max_memory_reserved() / 2**20, 2
                )
            except RuntimeError:
                pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        result["elapsed_seconds"] = round(time.monotonic() - started, 3)
        write_result(result, args.result_json)
        return 42 if is_oom else 1

    result["elapsed_seconds"] = round(time.monotonic() - started, 3)
    if not math.isfinite(float(result["loss"])):
        raise RuntimeError("probe produced a non-finite recorded loss")
    write_result(result, args.result_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

