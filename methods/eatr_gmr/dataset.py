"""Soccer-GMR feature dataset for isolated EaTR training and evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from .spans import span_xx_to_cxw


VIDEO_EXTENSIONS = {".mp4", ".mkv", ".webm", ".avi", ".mov", ".m4v"}


def video_id_to_feature_stem(video_id: str) -> str:
    """Map Soccer annotation ids such as ``foo.mp4`` to ``foo.npz``."""
    path = Path(str(video_id))
    return path.stem if path.suffix.lower() in VIDEO_EXTENSIONS else str(video_id)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def _l2_normalize(array: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(array, axis=-1, keepdims=True)
    return array / np.maximum(norm, 1e-8)


class SoccerGMRDataset(Dataset):
    """Read standard SlowFast, CLIP-video, and CLIP-text NPZ features.

    Required annotation fields are ``qid``, ``query``, ``vid``, and
    ``duration``. During training/evaluation with labels,
    ``relevant_windows`` may be a non-empty list or the empty list.
    """

    def __init__(
        self,
        annotation_path: str | Path,
        slowfast_dir: str | Path,
        clip_dir: str | Path,
        text_dir: str | Path,
        *,
        max_video_len: int = 75,
        max_text_len: int = 32,
        clip_length: float = 2.0,
        max_windows: int = 8,
        load_labels: bool = True,
        trim_text_by_attention_mask: bool = True,
        expected_video_feature_dim: int | None = 2816,
        expected_text_feature_dim: int | None = 512,
    ) -> None:
        self.annotation_path = Path(annotation_path)
        self.slowfast_dir = Path(slowfast_dir)
        self.clip_dir = Path(clip_dir)
        self.text_dir = Path(text_dir)
        self.max_video_len = int(max_video_len)
        self.max_text_len = int(max_text_len)
        self.clip_length = float(clip_length)
        self.max_windows = int(max_windows)
        self.load_labels = bool(load_labels)
        self.trim_text_by_attention_mask = bool(trim_text_by_attention_mask)
        self.expected_video_feature_dim = expected_video_feature_dim
        self.expected_text_feature_dim = expected_text_feature_dim
        self.data = _load_jsonl(self.annotation_path)
        if not self.data:
            raise ValueError(f"empty annotation file: {self.annotation_path}")

    def __len__(self) -> int:
        return len(self.data)

    def _feature_paths(self, row: dict[str, Any]) -> tuple[Path, Path, Path]:
        stem = video_id_to_feature_stem(str(row["vid"]))
        qid = row["qid"]
        return (
            self.slowfast_dir / f"{stem}.npz",
            self.clip_dir / f"{stem}.npz",
            self.text_dir / f"qid{qid}.npz",
        )

    @staticmethod
    def _read_npz(path: Path, key: str) -> tuple[np.ndarray, np.ndarray | None]:
        if not path.is_file():
            raise FileNotFoundError(f"missing feature: {path}")
        with np.load(path) as archive:
            if key not in archive:
                raise KeyError(f"{path}: missing NPZ key {key!r}")
            value = np.asarray(archive[key], dtype=np.float32)
            attention_mask = (
                np.asarray(archive["attention_mask"]) if "attention_mask" in archive else None
            )
        return value, attention_mask

    def _load_video(self, row: dict[str, Any]) -> torch.Tensor:
        slowfast_path, clip_path, _ = self._feature_paths(row)
        slowfast, _ = self._read_npz(slowfast_path, "features")
        clip, _ = self._read_npz(clip_path, "features")
        if slowfast.ndim != 2 or clip.ndim != 2:
            raise ValueError("video features must have shape [clips, channels]")
        length = min(len(slowfast), len(clip), self.max_video_len)
        if length < 1:
            raise ValueError(f"empty video features for vid={row['vid']}")
        video = np.concatenate(
            [_l2_normalize(slowfast[:length]), _l2_normalize(clip[:length])], axis=-1
        ).astype(np.float32, copy=False)
        if (
            self.expected_video_feature_dim is not None
            and video.shape[-1] != self.expected_video_feature_dim
        ):
            raise ValueError(
                f"vid={row['vid']}: expected {self.expected_video_feature_dim} video channels, "
                f"got {video.shape[-1]}"
            )
        start = np.arange(length, dtype=np.float32) / float(length)
        temporal_endpoint = np.stack([start, start + 1.0 / float(length)], axis=-1)
        return torch.from_numpy(np.concatenate([video, temporal_endpoint], axis=-1))

    def _load_text(self, row: dict[str, Any]) -> torch.Tensor:
        _, _, text_path = self._feature_paths(row)
        text, attention_mask = self._read_npz(text_path, "last_hidden_state")
        if text.ndim == 3 and text.shape[0] == 1:
            text = text[0]
        if text.ndim != 2:
            raise ValueError(f"{text_path}: text features must have shape [tokens, channels]")
        if self.trim_text_by_attention_mask and attention_mask is not None:
            attention_mask = np.asarray(attention_mask).reshape(-1)
            if len(attention_mask) != len(text):
                raise ValueError(f"{text_path}: feature/mask length mismatch")
            text = text[attention_mask > 0]
        text = text[: self.max_text_len]
        if len(text) < 1:
            raise ValueError(f"empty text features for qid={row['qid']}")
        if (
            self.expected_text_feature_dim is not None
            and text.shape[-1] != self.expected_text_feature_dim
        ):
            raise ValueError(
                f"qid={row['qid']}: expected {self.expected_text_feature_dim} text channels, "
                f"got {text.shape[-1]}"
            )
        return torch.from_numpy(_l2_normalize(text).astype(np.float32, copy=False))

    def _span_labels(self, windows: Iterable, video_length: int) -> torch.Tensor:
        values = list(windows or [])[: self.max_windows]
        if not values:
            return torch.zeros((0, 2), dtype=torch.float32)
        spans = torch.as_tensor(values, dtype=torch.float32).reshape(-1, 2)
        normalizer = float(video_length) * self.clip_length
        spans = (spans / normalizer).clamp(0.0, 1.0)
        return span_xx_to_cxw(spans)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.data[index]
        for field in ("qid", "query", "vid", "duration"):
            if field not in row:
                raise KeyError(f"{self.annotation_path}: row {index} missing {field!r}")
        video = self._load_video(row)
        sample = {
            "meta": row,
            "video": video,
            "text": self._load_text(row),
        }
        if self.load_labels:
            windows = row.get("relevant_windows", [])
            if windows is None:
                windows = []
            if not isinstance(windows, list):
                raise TypeError(f"qid={row['qid']}: relevant_windows must be a list")
            sample["spans"] = self._span_labels(windows, len(video))
            sample["exist_label"] = float(bool(windows))
            sample["count_label"] = min(len(windows), 4)
            sample["raw_count_label"] = len(windows)
        return sample


def _pad_features(values: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    max_length = max(len(value) for value in values)
    channels = values[0].shape[-1]
    padded = values[0].new_zeros((len(values), max_length, channels))
    mask = values[0].new_zeros((len(values), max_length))
    for index, value in enumerate(values):
        if value.shape[-1] != channels:
            raise ValueError("feature dimensions differ within one batch")
        padded[index, : len(value)] = value
        mask[index, : len(value)] = 1
    return padded, mask


def collate_fn(samples: list[dict[str, Any]]):
    video, video_mask = _pad_features([sample["video"] for sample in samples])
    text, text_mask = _pad_features([sample["text"] for sample in samples])
    model_inputs = {
        "src_vid": video,
        "src_vid_mask": video_mask,
        "src_txt": text,
        "src_txt_mask": text_mask,
    }
    targets = None
    if "spans" in samples[0]:
        targets = {
            "span_labels": [{"spans": sample["spans"]} for sample in samples],
            "exist_label": torch.tensor(
                [sample["exist_label"] for sample in samples], dtype=torch.float32
            ),
            "count_label": torch.tensor(
                [sample["count_label"] for sample in samples], dtype=torch.long
            ),
            "raw_count_label": torch.tensor(
                [sample["raw_count_label"] for sample in samples], dtype=torch.long
            ),
        }
    return [sample["meta"] for sample in samples], model_inputs, targets


def move_batch(model_inputs: dict, targets: dict | None, device: torch.device):
    model_inputs = {name: value.to(device) for name, value in model_inputs.items()}
    if targets is None:
        return model_inputs, None
    moved_targets = {
        "span_labels": [
            {"spans": target["spans"].to(device)} for target in targets["span_labels"]
        ],
        "exist_label": targets["exist_label"].to(device),
        "count_label": targets["count_label"].to(device),
        "raw_count_label": targets["raw_count_label"].to(device),
    }
    return model_inputs, moved_targets
