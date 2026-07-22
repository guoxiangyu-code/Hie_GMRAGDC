"""CG-specific HieA2M components without duplicate sentence/dummy pathways.

CG-DETR already implements sentence and dummy-token ACA.  The shared
DualGround-inspired module also has a sentence/dummy branch, so stacking the
whole module would duplicate semantics and parameters.  This file migrates
only RPG, Slot Attention, phrase EOS, and temporal phrase-video interaction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from models.moment_detr_gmr.dual_grounding import (
    RecurrentPhraseGenerator,
    SlotRefinement,
)


def _last_valid(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    lengths = mask.long().sum(dim=1).clamp_min(1)
    batch = torch.arange(tokens.shape[0], device=tokens.device)
    return tokens[batch, lengths - 1]


def _word_mask(mask: torch.Tensor) -> torch.Tensor:
    """Exclude CLIP SOT/EOS while retaining a finite degenerate fallback."""
    result = mask.bool().clone()
    if result.shape[1] == 0:
        return result
    result[:, 0] = False
    lengths = mask.long().sum(dim=1).clamp_min(1)
    batch = torch.arange(mask.shape[0], device=mask.device)
    result[batch, lengths - 1] = False
    empty = ~result.any(dim=1)
    if empty.any():
        result[batch[empty], lengths[empty] - 1] = True
    return result


@dataclass
class CGPhraseGroundingOutput:
    video: torch.Tensor
    phrase_attention: torch.Tensor
    phrase_eos: torch.Tensor
    text_eos: torch.Tensor
    phrase_gate: torch.Tensor


class CGPhraseGrounding(nn.Module):
    """DualGround phrase path adapted to CG's projected video/text tokens."""

    def __init__(
        self,
        hidden_dim: int,
        nheads: int = 8,
        dropout: float = 0.1,
        num_phrases: int = 3,
        max_text_len: int = 77,
        slot_iterations: int = 1,
        gate_init: float = -4.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_phrases = int(num_phrases)
        self.rpg = RecurrentPhraseGenerator(hidden_dim, num_phrases, max_text_len)
        self.slot_refine = SlotRefinement(hidden_dim, iterations=slot_iterations)
        self.phrase_eos_token = nn.Parameter(torch.empty(1, 1, hidden_dim))
        nn.init.normal_(self.phrase_eos_token, std=0.02)

        set_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nheads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.phrase_set_encoder = nn.TransformerEncoder(set_layer, num_layers=1)
        self.phrase_proj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU())
        self.video_proj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU())
        self.context_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nheads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.phrase_temporal = nn.TransformerEncoder(temporal_layer, num_layers=1)
        self.phrase_weight_q = nn.Linear(hidden_dim, hidden_dim)
        self.phrase_weight_k = nn.Linear(hidden_dim, hidden_dim)

        # Exact zero residual at initialization, with a nonzero derivative.
        self.gate_origin = float(gate_init)
        self.phrase_gate_logit = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(
        self,
        video: torch.Tensor,
        video_mask: torch.Tensor,
        text: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> CGPhraseGroundingOutput:
        video_mask = video_mask.bool()
        text_mask = text_mask.bool()
        eos = _last_valid(text, text_mask)
        lexical_mask = _word_mask(text_mask)
        initial, _ = self.rpg(text, lexical_mask, eos)
        phrases, phrase_attention = self.slot_refine(initial, text, lexical_mask)

        phrase_eos = self.phrase_eos_token.expand(text.shape[0], -1, -1)
        phrase_set = self.phrase_set_encoder(torch.cat([phrases, phrase_eos], dim=1))
        phrases = phrase_set[:, : self.num_phrases]
        phrase_eos = phrase_set[:, self.num_phrases]

        phrase_features = self.phrase_proj(phrases)[:, :, None, :]
        video_features = self.video_proj(video)[:, None, :, :]
        context = self.context_proj(phrase_features * video_features)
        batch, num_phrases, length, dim = context.shape
        flat_context = context.reshape(batch * num_phrases, length, dim)
        flat_mask = (~video_mask)[:, None].expand(-1, num_phrases, -1)
        flat_context = self.phrase_temporal(
            flat_context,
            src_key_padding_mask=flat_mask.reshape(batch * num_phrases, length),
        )
        context = flat_context.reshape(batch, num_phrases, length, dim)

        phrase_logits = torch.einsum(
            "bd,bnd->bn",
            self.phrase_weight_q(phrase_eos),
            self.phrase_weight_k(phrases),
        ) / math.sqrt(self.hidden_dim)
        phrase_weights = F.softmax(phrase_logits, dim=1)
        phrase_video = torch.einsum("bn,bntd->btd", phrase_weights, context)
        phrase_video = phrase_video * video_mask[..., None].to(phrase_video.dtype)

        gate = torch.tanh(self.phrase_gate_logit - self.gate_origin)
        fused = video + gate * phrase_video
        fused = fused * video_mask[..., None].to(fused.dtype)
        return CGPhraseGroundingOutput(
            video=fused,
            phrase_attention=phrase_attention,
            phrase_eos=phrase_eos,
            text_eos=eos,
            phrase_gate=gate,
        )
