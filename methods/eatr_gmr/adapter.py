"""Paper-style existence adapter for Generalized Moment Retrieval."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class GMRExistenceAdapter(nn.Module):
    """Max-pool decoder slots and predict whether any moment exists.

    This implements Equations (6)--(8) of the Soccer-GMR paper: the final
    cross-modal moment queries are max pooled over the query dimension and
    passed through a two-layer MLP with a ReLU hidden activation.
    """

    def __init__(self, input_dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden_dim = input_dim if hidden_dim is None else hidden_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, decoder_queries: torch.Tensor) -> torch.Tensor:
        if decoder_queries.ndim != 3:
            raise ValueError(
                "decoder_queries must have shape [batch, queries, hidden], "
                f"got {tuple(decoder_queries.shape)}"
            )
        pooled = decoder_queries.max(dim=1).values
        return self.fc2(F.relu(self.fc1(pooled))).squeeze(-1)


def existence_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Binary cross entropy used by the parallel GMR branch."""
    return F.binary_cross_entropy_with_logits(
        logits.reshape(-1), labels.to(dtype=logits.dtype).reshape(-1)
    )
