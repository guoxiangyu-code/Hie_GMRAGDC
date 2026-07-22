"""Argument helpers shared by the train and evaluate module CLIs."""

from __future__ import annotations

import argparse


def add_data_arguments(parser: argparse.ArgumentParser, *, require_train: bool = False) -> None:
    annotation_name = "--train-annotations" if require_train else "--annotations"
    parser.add_argument(annotation_name, required=True)
    parser.add_argument("--slowfast-dir", required=True)
    parser.add_argument("--clip-dir", required=True)
    parser.add_argument("--text-dir", required=True)
    parser.add_argument("--max-video-len", type=int, default=75)
    parser.add_argument("--max-text-len", type=int, default=32)
    parser.add_argument("--clip-length", type=float, default=2.0)
    parser.add_argument("--max-windows", type=int, default=8)
    parser.add_argument(
        "--trim-text-by-attention-mask",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "remove CLIP padding tokens; disabled by default for the released "
            "Soccer-GMR 32-token protocol"
        ),
    )
    parser.add_argument(
        "--video-feature-dim", type=int, default=2816,
        help="SlowFast+CLIP channels before the two temporal-endpoint channels",
    )
    parser.add_argument("--text-feature-dim", type=int, default=512)


def add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--nheads", type=int, default=8)
    parser.add_argument("--enc-layers", type=int, default=3)
    parser.add_argument("--dec-layers", type=int, default=3)
    parser.add_argument("--dim-feedforward", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--input-dropout", type=float, default=0.5)
    parser.add_argument("--num-queries", type=int, default=10)
    parser.add_argument("--num-slot-iter", type=int, default=3)
    parser.add_argument("--n-input-proj", type=int, default=2)
    parser.add_argument("--use-text-position", action="store_true")
    parser.add_argument("--no-aux-loss", action="store_true")


def add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2018)
