"""Soccer-GMR feature dataset used by the isolated CG-DETR runner."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .span_utils import span_xx_to_cxw


VIDEO_EXTENSIONS = (".mp4", ".mkv", ".webm", ".avi", ".mov", ".m4v")


def video_id_to_feature_stem(video_id: str) -> str:
    """Map annotation video names such as ``foo.mp4`` to feature stem ``foo``."""
    lowered = video_id.lower()
    for extension in VIDEO_EXTENSIONS:
        if lowered.endswith(extension):
            return video_id[: -len(extension)]
    return video_id


def _l2_normalize(array: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(array, axis=-1, keepdims=True)
    return array / np.maximum(denom, 1e-12)


def _load_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class SoccerGMRDataset(Dataset):
    """Load CLIP+SlowFast video features and token-level CLIP text features.

    ``sample_mode`` is part of the protocol rather than hidden filtering:
    ``positive`` reproduces a conventional MR baseline, ``mixed`` retains both
    positive and null queries for GMR, and ``null`` exists for regression tests.
    """

    def __init__(
        self,
        annotation_path: str | Path,
        video_feature_dirs: Sequence[str | Path],
        text_feature_dir: str | Path,
        *,
        max_q_l: int = 32,
        max_v_l: int = 75,
        max_windows: int = 10,
        clip_length: float = 2.0,
        use_tef: bool = True,
        trim_text_by_attention_mask: bool = False,
        sample_mode: str = "mixed",
        max_samples: int | None = None,
    ) -> None:
        if sample_mode not in {"positive", "mixed", "null"}:
            raise ValueError(f"unknown sample_mode={sample_mode!r}")
        self.annotation_path = Path(annotation_path)
        self.video_feature_dirs = [Path(path) for path in video_feature_dirs]
        self.text_feature_dir = Path(text_feature_dir)
        self.max_q_l = max_q_l
        self.max_v_l = max_v_l
        self.max_windows = max_windows
        self.clip_length = clip_length
        self.use_tef = use_tef
        self.trim_text_by_attention_mask = trim_text_by_attention_mask

        records = _load_jsonl(self.annotation_path)
        if sample_mode == "positive":
            records = [row for row in records if row.get("relevant_windows", [])]
        elif sample_mode == "null":
            records = [row for row in records if not row.get("relevant_windows", [])]
        if max_samples is not None:
            records = records[:max_samples]
        if not records:
            raise ValueError(f"no records selected from {annotation_path}")
        self.data = records

    def __len__(self) -> int:
        return len(self.data)

    def _resolve_video_feature(self, root: Path, video_id: str) -> Path:
        stem = video_id_to_feature_stem(video_id)
        candidates = [root / f"{stem}.npz"]
        if stem != video_id:
            # Some feature exports retain the media extension; support both
            # layouts while preferring the documented extension-free stem.
            candidates.append(root / f"{video_id}.npz")
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(", ".join(str(path) for path in candidates))

    def _load_video(self, video_id: str) -> torch.Tensor:
        streams = []
        for root in self.video_feature_dirs:
            with np.load(self._resolve_video_feature(root, video_id)) as archive:
                feature = archive["features"][: self.max_v_l].astype(np.float32)
            streams.append(_l2_normalize(feature))
        common_length = min(len(stream) for stream in streams)
        video = np.concatenate([stream[:common_length] for stream in streams], axis=-1)
        tensor = torch.from_numpy(video)
        if self.use_tef:
            starts = torch.arange(common_length, dtype=torch.float32) / common_length
            tef = torch.stack([starts, starts + 1.0 / common_length], dim=-1)
            tensor = torch.cat([tensor, tef], dim=-1)
        return tensor

    def _load_text(self, qid: int | str) -> torch.Tensor:
        path = self.text_feature_dir / f"qid{qid}.npz"
        with np.load(path) as archive:
            feature = archive["last_hidden_state"].astype(np.float32)
            attention_mask = archive.get("attention_mask")
        if self.trim_text_by_attention_mask and attention_mask is not None:
            valid = np.asarray(attention_mask).reshape(-1) > 0
            if len(valid) != len(feature) or not valid.any():
                raise ValueError(f"invalid attention mask in {path}")
            feature = feature[valid]
        return torch.from_numpy(_l2_normalize(feature[: self.max_q_l]))

    def _span_labels(self, windows: Iterable[Sequence[float]], context_length: int) -> torch.Tensor:
        windows = list(windows)
        if not windows:
            return torch.zeros((0, 2), dtype=torch.float32)
        if len(windows) > self.max_windows:
            # Match the upstream training behavior while avoiding mutation of
            # the annotation record itself.
            windows = random.sample(windows, self.max_windows)
        normalized = torch.tensor(windows, dtype=torch.float32)
        normalized = normalized / (context_length * self.clip_length)
        return span_xx_to_cxw(normalized)

    def _relevant_clips(
        self, windows: Iterable[Sequence[float]], context_length: int
    ) -> torch.Tensor:
        """Build a union-of-windows clip mask with half-open interval overlap.

        Unlike QVHighlights, Soccer-GMR has no per-clip saliency ratings.  CG's
        moment-aware auxiliary path only needs the binary support of the MR
        windows, which is available without fabricating QV labels.  Every
        annotated window contributes; a null query is exactly all zeros.
        """
        mask = torch.zeros(context_length, dtype=torch.float32)
        for window in windows:
            if len(window) != 2:
                raise ValueError(f"invalid relevant window: {window!r}")
            start = max(0.0, min(float(window[0]), context_length * self.clip_length))
            end = max(0.0, min(float(window[1]), context_length * self.clip_length))
            if end <= start:
                continue
            clip_starts = torch.arange(context_length, dtype=torch.float32) * self.clip_length
            clip_ends = clip_starts + self.clip_length
            mask[(clip_starts < end) & (clip_ends > start)] = 1.0
        return mask

    def __getitem__(self, index: int) -> dict:
        meta = self.data[index]
        video = self._load_video(str(meta["vid"]))
        windows = meta.get("relevant_windows", []) or []
        return {
            "meta": meta,
            "model_inputs": {
                "query_feat": self._load_text(meta["qid"]),
                "video_feat": video,
                "span_labels": self._span_labels(windows, len(video)),
                "relevant_clips": self._relevant_clips(windows, len(video)),
                "exist_label": float(bool(windows)),
                "count_label": min(len(windows), 4),
                "raw_count_label": len(windows),
            },
        }


def _pad_1d(sequences: Sequence[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    max_length = max(len(sequence) for sequence in sequences)
    feature_shape = sequences[0].shape[1:]
    padded = sequences[0].new_zeros((len(sequences), max_length, *feature_shape))
    mask = torch.zeros((len(sequences), max_length), dtype=torch.float32)
    for index, sequence in enumerate(sequences):
        padded[index, : len(sequence)] = sequence
        mask[index, : len(sequence)] = 1.0
    return padded, mask


def collate_fn(batch: Sequence[dict]) -> tuple[list[dict], dict]:
    return [item["meta"] for item in batch], {
        "query_feat": _pad_1d([item["model_inputs"]["query_feat"] for item in batch]),
        "video_feat": _pad_1d([item["model_inputs"]["video_feat"] for item in batch]),
        "span_labels": [
            {"spans": item["model_inputs"]["span_labels"]} for item in batch
        ],
        "relevant_clips": _pad_1d(
            [item["model_inputs"]["relevant_clips"] for item in batch]
        )[0],
        "exist_label": torch.tensor(
            [item["model_inputs"]["exist_label"] for item in batch], dtype=torch.float32
        ),
        "count_label": torch.tensor(
            [item["model_inputs"]["count_label"] for item in batch], dtype=torch.long
        ),
        "raw_count_label": torch.tensor(
            [item["model_inputs"]["raw_count_label"] for item in batch], dtype=torch.long
        ),
    }


def prepare_batch(batch: tuple[list[dict], dict], device: torch.device) -> tuple[list[dict], dict, dict]:
    metadata, collated = batch
    query, query_mask = collated["query_feat"]
    video, video_mask = collated["video_feat"]
    inputs = {
        "src_txt": query.to(device),
        "src_txt_mask": query_mask.to(device),
        "src_vid": video.to(device),
        "src_vid_mask": video_mask.to(device),
    }
    targets = {
        "span_labels": [
            {"spans": item["spans"].to(device)} for item in collated["span_labels"]
        ],
        "exist_label": collated["exist_label"].to(device),
        "positive_mask": collated["exist_label"].to(device).bool(),
        "relevant_clips": collated["relevant_clips"].to(device),
        "count_label": collated["count_label"].to(device),
        "raw_count_label": collated["raw_count_label"].to(device),
    }
    return metadata, inputs, targets
