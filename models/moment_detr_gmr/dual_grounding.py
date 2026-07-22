"""DualGround-inspired sentence/phrase temporal feature interaction.

The original DualGround decoder is convolutional.  This module keeps its two
semantic paths, but returns video tokens that can be consumed by any DETR-style
moment decoder:

* the last valid CLIP token (EOS) drives a dummy-enhanced sentence path;
* lexical tokens are grouped into recurrent, slot-refined phrase units;
* sentence- and phrase-conditioned video features are injected through small
  learnable residual gates.

No phrase labels or Soccer-GMR metadata are used by the forward pass.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


def _last_valid(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Gather the last valid token for each item in a padded batch."""
    lengths = mask.long().sum(dim=1).clamp_min(1)
    batch = torch.arange(x.shape[0], device=x.device)
    return x[batch, lengths - 1]


def _safe_word_mask(mask: torch.Tensor) -> torch.Tensor:
    """Return valid lexical positions, excluding CLIP SOT and EOS tokens."""
    word_mask = mask.bool().clone()
    if word_mask.shape[1] == 0:
        return word_mask
    word_mask[:, 0] = False
    lengths = mask.long().sum(dim=1).clamp_min(1)
    batch = torch.arange(mask.shape[0], device=mask.device)
    word_mask[batch, lengths - 1] = False

    # Degenerate one/two-token inputs still need a finite attention row.  This
    # fallback uses the last valid token only for numerical safety.
    empty = ~word_mask.any(dim=1)
    if empty.any():
        word_mask[batch[empty], lengths[empty] - 1] = True
    return word_mask


class RecurrentPhraseGenerator(nn.Module):
    """Generate fixed-count phrase slots following DualGround RPG (Eq. 2)."""

    def __init__(self, hidden_dim: int, num_phrases: int, max_text_len: int):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_phrases = int(num_phrases)
        self.position = nn.Embedding(int(max_text_len), hidden_dim)
        self.guides = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(num_phrases)
        ])
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        words: torch.Tensor,
        word_mask: torch.Tensor,
        eos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, length, _ = words.shape
        positions = torch.arange(length, device=words.device)
        positioned = words + self.position(positions)[None]
        keys = self.key(positioned)
        values = self.value(words)

        previous = torch.zeros_like(eos)
        phrases = []
        attentions = []
        for guide_layer in self.guides:
            guide = guide_layer(torch.cat([eos, previous], dim=-1))
            logits = torch.einsum("bd,bld->bl", self.query(guide), keys)
            logits = logits / math.sqrt(self.hidden_dim)
            logits = logits.masked_fill(~word_mask, torch.finfo(logits.dtype).min)
            attention = F.softmax(logits, dim=-1)
            phrase = torch.einsum("bl,bld->bd", attention, values)
            phrases.append(phrase)
            attentions.append(attention)
            previous = phrase

        return torch.stack(phrases, dim=1), torch.stack(attentions, dim=1)


class SlotRefinement(nn.Module):
    """A compact Slot-Attention refinement initialized by RPG phrases."""

    def __init__(self, hidden_dim: int, iterations: int = 1, eps: float = 1e-6):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.iterations = int(iterations)
        self.eps = float(eps)
        self.norm_words = nn.LayerNorm(hidden_dim)
        self.norm_slots = nn.LayerNorm(hidden_dim)
        self.norm_mlp = nn.LayerNorm(hidden_dim)
        self.to_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.to_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.to_v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(
        self,
        slots: torch.Tensor,
        words: torch.Tensor,
        word_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, num_slots, dim = slots.shape
        keys = self.to_k(self.norm_words(words))
        values = self.to_v(self.norm_words(words))
        final_attention = None

        for _ in range(self.iterations):
            previous = slots
            queries = self.to_q(self.norm_slots(slots))
            logits = torch.einsum("bnd,bld->bnl", queries, keys)
            logits = logits / math.sqrt(self.hidden_dim)
            logits = logits.masked_fill(
                ~word_mask[:, None], torch.finfo(logits.dtype).min
            )

            # Slot-wise competition for each word, followed by input-wise
            # normalization, as used by Slot Attention.
            attention = F.softmax(logits, dim=1)
            valid_words = word_mask[:, None].to(attention.dtype)
            attention = (attention + self.eps) * valid_words
            attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(self.eps)
            updates = torch.einsum("bnl,bld->bnd", attention, values)
            slots = self.gru(
                updates.reshape(bsz * num_slots, dim),
                previous.reshape(bsz * num_slots, dim),
            ).reshape(bsz, num_slots, dim)
            slots = slots + self.mlp(self.norm_mlp(slots))
            final_attention = attention

        assert final_attention is not None
        return slots, final_attention


@dataclass
class DualGroundingOutput:
    video: torch.Tensor
    phrase_attention: torch.Tensor
    phrase_eos: torch.Tensor
    text_eos: torch.Tensor
    sentence_eos_attention: torch.Tensor
    sentence_gate: torch.Tensor
    phrase_gate: torch.Tensor


class TemporalDualGrounding(nn.Module):
    """Create sentence- and phrase-aware residuals for projected video tokens."""

    def __init__(
        self,
        hidden_dim: int,
        nheads: int = 8,
        dropout: float = 0.1,
        num_phrases: int = 3,
        num_dummies: int = 3,
        max_text_len: int = 77,
        phrase_slot_iterations: int = 1,
        gate_init: float = -4.0,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_phrases = int(num_phrases)
        self.num_dummies = int(num_dummies)

        self.dummy_tokens = nn.Parameter(torch.empty(num_dummies, hidden_dim))
        nn.init.normal_(self.dummy_tokens, std=0.02)
        dummy_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nheads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.dummy_encoder = nn.TransformerEncoder(dummy_layer, num_layers=1)
        self.sentence_q = nn.Linear(hidden_dim, hidden_dim)
        self.sentence_k = nn.Linear(hidden_dim, hidden_dim)
        self.sentence_v = nn.Linear(hidden_dim, hidden_dim)
        sentence_temporal_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nheads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.sentence_temporal = nn.TransformerEncoder(sentence_temporal_layer, num_layers=1)

        self.rpg = RecurrentPhraseGenerator(hidden_dim, num_phrases, max_text_len)
        self.slot_refine = SlotRefinement(hidden_dim, iterations=phrase_slot_iterations)
        self.phrase_eos_token = nn.Parameter(torch.empty(1, 1, hidden_dim))
        nn.init.normal_(self.phrase_eos_token, std=0.02)
        phrase_set_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nheads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.phrase_set_encoder = nn.TransformerEncoder(phrase_set_layer, num_layers=1)
        self.phrase_proj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU())
        self.video_proj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU())
        self.context_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        phrase_temporal_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nheads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.phrase_temporal = nn.TransformerEncoder(phrase_temporal_layer, num_layers=1)
        self.phrase_weight_q = nn.Linear(hidden_dim, hidden_dim)
        self.phrase_weight_k = nn.Linear(hidden_dim, hidden_dim)

        # A zero-centered parameterization makes the residual exactly zero at
        # warm-start, while tanh'(0)=1 keeps a strong gradient for each gate.
        self.gate_origin = float(gate_init)
        self.sentence_gate_logit = nn.Parameter(torch.tensor(float(gate_init)))
        self.phrase_gate_logit = nn.Parameter(torch.tensor(float(gate_init)))

    def _sentence_path(
        self,
        video: torch.Tensor,
        video_mask: torch.Tensor,
        eos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz = video.shape[0]
        dummies = self.dummy_tokens[None].expand(bsz, -1, -1)
        encoded = self.dummy_encoder(torch.cat([dummies, eos[:, None]], dim=1))
        # DualGround keeps the contextualized dummies but the original EOS.
        keys_values = torch.cat([encoded[:, : self.num_dummies], eos[:, None]], dim=1)
        queries = self.sentence_q(video)
        keys = self.sentence_k(keys_values)
        values = self.sentence_v(keys_values)
        logits = torch.einsum("btd,bkd->btk", queries, keys) / math.sqrt(self.hidden_dim)
        attention = F.softmax(logits, dim=-1)
        eos_attention = attention[..., -1]
        sentence = eos_attention[..., None] * values[:, None, -1]
        sentence = self.sentence_temporal(
            sentence,
            src_key_padding_mask=~video_mask.bool(),
        )
        sentence = sentence * video_mask[..., None].to(sentence.dtype)
        return sentence, eos_attention

    def _phrase_path(
        self,
        video: torch.Tensor,
        video_mask: torch.Tensor,
        text: torch.Tensor,
        word_mask: torch.Tensor,
        eos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        initial, _ = self.rpg(text, word_mask, eos)
        phrases, phrase_attention = self.slot_refine(initial, text, word_mask)
        phrase_eos = self.phrase_eos_token.expand(text.shape[0], -1, -1)
        phrase_set = self.phrase_set_encoder(torch.cat([phrases, phrase_eos], dim=1))
        phrases = phrase_set[:, : self.num_phrases]
        phrase_eos = phrase_set[:, self.num_phrases]

        phrase_features = self.phrase_proj(phrases)[:, :, None, :]
        video_features = self.video_proj(video)[:, None, :, :]
        context = self.context_proj(phrase_features * video_features)

        bsz, num_phrases, length, dim = context.shape
        flat_context = context.reshape(bsz * num_phrases, length, dim)
        flat_mask = (~video_mask.bool())[:, None].expand(-1, num_phrases, -1)
        flat_context = self.phrase_temporal(
            flat_context,
            src_key_padding_mask=flat_mask.reshape(bsz * num_phrases, length),
        )
        context = flat_context.reshape(bsz, num_phrases, length, dim)

        phrase_logits = torch.einsum(
            "bd,bnd->bn",
            self.phrase_weight_q(phrase_eos),
            self.phrase_weight_k(phrases),
        ) / math.sqrt(self.hidden_dim)
        phrase_weights = F.softmax(phrase_logits, dim=1)
        phrase_video = torch.einsum("bn,bntd->btd", phrase_weights, context)
        phrase_video = phrase_video * video_mask[..., None].to(phrase_video.dtype)
        return phrase_video, phrase_attention, phrase_eos

    def forward(
        self,
        video: torch.Tensor,
        video_mask: torch.Tensor,
        text: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> DualGroundingOutput:
        text_mask = text_mask.bool()
        video_mask = video_mask.bool()
        eos = _last_valid(text, text_mask)
        word_mask = _safe_word_mask(text_mask)
        sentence_video, eos_attention = self._sentence_path(video, video_mask, eos)
        phrase_video, phrase_attention, phrase_eos = self._phrase_path(
            video, video_mask, text, word_mask, eos
        )
        sentence_gate = torch.tanh(self.sentence_gate_logit - self.gate_origin)
        phrase_gate = torch.tanh(self.phrase_gate_logit - self.gate_origin)
        fused = video + sentence_gate * sentence_video + phrase_gate * phrase_video
        fused = fused * video_mask[..., None].to(fused.dtype)
        return DualGroundingOutput(
            video=fused,
            phrase_attention=phrase_attention,
            phrase_eos=phrase_eos,
            text_eos=eos,
            sentence_eos_attention=eos_attention,
            sentence_gate=sentence_gate,
            phrase_gate=phrase_gate,
        )


def dual_grounding_losses(
    outputs: dict[str, torch.Tensor],
    dqa_scale: float = 1.0,
    temperature: float = 0.07,
    sample_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """DualGround DQA (Eq. 5) and symmetric EOS reconstruction losses."""
    attention = outputs["dual_phrase_attention"]
    phrase_eos = outputs["dual_phrase_eos"]
    text_eos = outputs["dual_text_eos"].detach()
    if sample_mask is not None:
        sample_mask = sample_mask.to(device=attention.device).reshape(-1).bool()
        if sample_mask.numel() != attention.shape[0]:
            raise ValueError(
                "DualGround sample_mask batch size does not match outputs"
            )
        if not sample_mask.any():
            zero = attention.sum() * 0.0 + phrase_eos.sum() * 0.0
            return {"loss_dual_dqa": zero, "loss_dual_eos": zero}
        attention = attention[sample_mask]
        phrase_eos = phrase_eos[sample_mask]
        text_eos = text_eos[sample_mask]
    phrase_eos = F.normalize(phrase_eos, dim=-1)
    text_eos = F.normalize(text_eos, dim=-1)

    gram = torch.bmm(attention, attention.transpose(1, 2))
    identity = torch.eye(gram.shape[-1], device=gram.device, dtype=gram.dtype)[None]
    loss_dqa = (gram - float(dqa_scale) * identity).pow(2).sum(dim=(1, 2)).mean()

    # Pairwise reconstruction keeps the objective meaningful for a one-item
    # smoke batch; the symmetric InfoNCE term adds cross-sample discrimination.
    loss_pair = (1.0 - (phrase_eos * text_eos).sum(dim=-1)).mean()
    if phrase_eos.shape[0] > 1:
        logits = phrase_eos @ text_eos.transpose(0, 1) / float(temperature)
        labels = torch.arange(logits.shape[0], device=logits.device)
        loss_nce = 0.5 * (
            F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)
        )
    else:
        loss_nce = phrase_eos.sum() * 0.0
    return {"loss_dual_dqa": loss_dqa, "loss_dual_eos": loss_pair + loss_nce}
