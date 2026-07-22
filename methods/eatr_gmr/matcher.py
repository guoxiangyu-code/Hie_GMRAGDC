"""Null-safe Hungarian matchers adapted from the official EaTR implementation."""

from __future__ import annotations

import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from .spans import generalized_temporal_iou, generalized_temporal_iou_, span_cxw_to_xx


def _empty_indices(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    empty = torch.empty(0, dtype=torch.int64, device=device)
    return empty, empty.clone()


class HungarianMatcher(nn.Module):
    """Match moment slots independently per sample, including empty targets."""

    def __init__(self, cost_class: float = 4.0, cost_span: float = 10.0,
                 cost_giou: float = 1.0) -> None:
        super().__init__()
        if cost_class == cost_span == cost_giou == 0:
            raise ValueError("at least one matching cost must be non-zero")
        self.cost_class = float(cost_class)
        self.cost_span = float(cost_span)
        self.cost_giou = float(cost_giou)

    @torch.no_grad()
    def forward(self, outputs: dict, targets: dict):
        pred_spans = outputs["pred_spans"]
        pred_prob = outputs["pred_logits"].softmax(-1)
        target_list = targets["span_labels"]
        if len(target_list) != pred_spans.shape[0]:
            raise ValueError("target batch size does not match model output")

        assignments = []
        for sample_spans, sample_prob, target in zip(pred_spans, pred_prob, target_list):
            target_spans = target["spans"]
            if target_spans.ndim != 2 or target_spans.shape[-1] != 2:
                raise ValueError(f"target spans must be [N,2], got {tuple(target_spans.shape)}")
            if target_spans.shape[0] == 0:
                assignments.append(_empty_indices(sample_spans.device))
                continue

            cost_class = -sample_prob[:, 0, None].expand(-1, target_spans.shape[0])
            cost_span = torch.cdist(sample_spans, target_spans, p=1)
            cost_giou = -generalized_temporal_iou(
                span_cxw_to_xx(sample_spans), span_cxw_to_xx(target_spans)
            )
            cost = (
                self.cost_class * cost_class
                + self.cost_span * cost_span
                + self.cost_giou * cost_giou
            )
            src, dst = linear_sum_assignment(cost.detach().cpu())
            assignments.append((
                torch.as_tensor(src, dtype=torch.int64, device=sample_spans.device),
                torch.as_tensor(dst, dtype=torch.int64, device=sample_spans.device),
            ))
        return assignments


class HungarianEventMatcher(nn.Module):
    """Null-safe matcher for EaTR's feature-derived pseudo events."""

    def __init__(self, cost_span: float = 10.0, cost_giou: float = 1.0) -> None:
        super().__init__()
        if cost_span == cost_giou == 0:
            raise ValueError("at least one event matching cost must be non-zero")
        self.cost_span = float(cost_span)
        self.cost_giou = float(cost_giou)

    @torch.no_grad()
    def forward(self, outputs: torch.Tensor, targets: list[torch.Tensor]):
        assignments = []
        for sample_spans, target_spans in zip(outputs, targets):
            if target_spans.shape[0] == 0:
                assignments.append(_empty_indices(sample_spans.device))
                continue
            target_spans = target_spans.to(sample_spans)
            cost_span = torch.cdist(sample_spans, target_spans, p=1)
            cost_giou = -generalized_temporal_iou_(
                span_cxw_to_xx(sample_spans), span_cxw_to_xx(target_spans)
            )
            cost = self.cost_span * cost_span + self.cost_giou * cost_giou
            src, dst = linear_sum_assignment(cost.detach().cpu())
            assignments.append((
                torch.as_tensor(src, dtype=torch.int64, device=sample_spans.device),
                torch.as_tensor(dst, dtype=torch.int64, device=sample_spans.device),
            ))
        return assignments
