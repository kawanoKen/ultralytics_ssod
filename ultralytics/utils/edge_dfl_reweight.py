"""Shared edge-wise DFL selector and weighting primitives.

The functions here deliberately operate only on selector masks and per-edge DFL
losses.  They never alter pseudo labels, assignment, classification targets, or
box/IoU loss.  This makes Oracle and DFL-confidence selectors share exactly the
same weighting and normalization path.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class EdgeWeightResult:
    """Raw and effective weights for a ``(positive prediction, edge)`` tensor."""

    raw: torch.Tensor
    normalized: torch.Tensor
    normalization_zero_sum: bool


def clip_xyxy(boxes: torch.Tensor | np.ndarray, image_h: float | torch.Tensor, image_w: float | torch.Tensor):
    """Return XYXY boxes clipped to the image region without modifying the input."""
    out = boxes.clone() if isinstance(boxes, torch.Tensor) else boxes.astype(np.float32, copy=True)
    if isinstance(out, torch.Tensor):
        out[..., 0::2] = out[..., 0::2].clamp(0, image_w)
        out[..., 1::2] = out[..., 1::2].clamp(0, image_h)
    else:
        out[..., 0::2] = np.clip(out[..., 0::2], 0, image_w)
        out[..., 1::2] = np.clip(out[..., 1::2], 0, image_h)
    return out


def select_dfl_low_confidence_edges(edge_confidence: torch.Tensor, threshold: float) -> torch.Tensor:
    """Select only low-confidence DFL edges; shape is preserved as ``(..., 4)``."""
    if edge_confidence.shape[-1] != 4:
        raise ValueError(f"edge_confidence must end in 4 edges, got {tuple(edge_confidence.shape)}")
    return edge_confidence < threshold


def make_edge_dfl_weights(selected_edges: torch.Tensor, selected_weight: float, normalize: bool) -> EdgeWeightResult:
    """Build common raw/effective edge weights for dose-response experiments.

    With normalization, the mean effective weight across all eligible edges is
    one.  If ``selected_weight == 0`` and every eligible edge is selected, the
    effective weights remain zero and the caller must treat its DFL contribution
    as zero rather than changing the requested experiment.
    """
    if selected_edges.dtype != torch.bool or selected_edges.shape[-1] != 4:
        raise ValueError(f"selected_edges must be bool with final dim 4, got {selected_edges.dtype} {tuple(selected_edges.shape)}")
    if selected_weight < 0:
        raise ValueError(f"selected_weight must be non-negative, got {selected_weight}")
    raw = torch.ones_like(selected_edges, dtype=torch.float32)
    raw = torch.where(selected_edges, torch.as_tensor(selected_weight, dtype=raw.dtype, device=raw.device), raw)
    if not normalize or raw.numel() == 0:
        return EdgeWeightResult(raw=raw, normalized=raw, normalization_zero_sum=False)
    total = raw.sum()
    if float(total.detach()) == 0.0:
        return EdgeWeightResult(raw=raw, normalized=raw, normalization_zero_sum=True)
    normalized = raw * (raw.numel() / total)
    return EdgeWeightResult(raw=raw, normalized=normalized, normalization_zero_sum=False)


def dfl_per_edge_loss(pred_dist: torch.Tensor, target_ltrb: torch.Tensor, reg_max: int) -> torch.Tensor:
    """DFL cross-entropy before the four box edges are reduced; returns ``(N, 4)``."""
    target = target_ltrb.clamp(0, reg_max - 1 - 0.01)
    tl = target.long()
    tr = tl + 1
    wl = tr - target
    wr = 1 - wl
    logits = pred_dist.reshape(-1, reg_max)
    ce_l = F.cross_entropy(logits, tl.reshape(-1), reduction="none").view_as(target)
    ce_r = F.cross_entropy(logits, tr.reshape(-1), reduction="none").view_as(target)
    return ce_l * wl + ce_r * wr


def reduce_weighted_dfl(
    per_edge_loss: torch.Tensor,
    target_weight: torch.Tensor,
    target_scores_sum: torch.Tensor | float,
    edge_weights: torch.Tensor,
) -> torch.Tensor:
    """Apply edge weights before the four-edge mean, retaining the baseline denominator."""
    if per_edge_loss.shape != edge_weights.shape:
        raise ValueError(f"per-edge loss/weight shape mismatch: {tuple(per_edge_loss.shape)} vs {tuple(edge_weights.shape)}")
    per_box = (per_edge_loss * edge_weights.to(per_edge_loss.dtype)).mean(-1)
    return (per_box.unsqueeze(-1) * target_weight).sum() / target_scores_sum


@torch.no_grad()
def oracle_selected_edges(
    pseudo_boxes: torch.Tensor,
    pseudo_classes: torch.Tensor,
    pseudo_valid: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_classes: torch.Tensor,
    gt_valid: torch.Tensor,
    image_h: float | torch.Tensor,
    image_w: float | torch.Tensor,
    error_threshold: float,
    match_iou: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select wrong pseudo edges with deterministic same-class greedy GT matching.

    Inputs are padded per-image XYXY targets in the *student loss coordinate
    system*.  The output has one four-edge mask per pseudo object.  Unmatched
    pseudo objects remain all-False, so GT only influences the selector.
    """
    if pseudo_boxes.ndim != 3 or gt_boxes.ndim != 3 or pseudo_boxes.shape[-1] != 4 or gt_boxes.shape[-1] != 4:
        raise ValueError("pseudo_boxes and gt_boxes must be (B, N, 4) XYXY tensors")
    if pseudo_boxes.shape[0] != gt_boxes.shape[0]:
        raise ValueError("pseudo and GT batch dimensions must match")
    selected = torch.zeros_like(pseudo_boxes, dtype=torch.bool)
    matched = torch.zeros_like(pseudo_valid, dtype=torch.bool)
    pseudo_boxes = clip_xyxy(pseudo_boxes, image_h, image_w)
    gt_boxes = clip_xyxy(gt_boxes, image_h, image_w)
    pseudo_classes = pseudo_classes.squeeze(-1).long()
    gt_classes = gt_classes.squeeze(-1).long()

    for batch_idx in range(pseudo_boxes.shape[0]):
        pidx = torch.where(pseudo_valid[batch_idx].squeeze(-1))[0]
        gidx = torch.where(gt_valid[batch_idx].squeeze(-1))[0]
        if pidx.numel() == 0 or gidx.numel() == 0:
            continue
        pboxes, gboxes = pseudo_boxes[batch_idx, pidx], gt_boxes[batch_idx, gidx]
        lt = torch.maximum(pboxes[:, None, :2], gboxes[None, :, :2])
        rb = torch.minimum(pboxes[:, None, 2:], gboxes[None, :, 2:])
        inter = (rb - lt).clamp_min(0).prod(-1)
        parea = (pboxes[:, 2:] - pboxes[:, :2]).clamp_min(0).prod(-1)
        garea = (gboxes[:, 2:] - gboxes[:, :2]).clamp_min(0).prod(-1)
        iou = inter / (parea[:, None] + garea[None, :] - inter).clamp_min(1e-9)
        same_class = pseudo_classes[batch_idx, pidx][:, None] == gt_classes[batch_idx, gidx][None, :]
        scores = torch.where(same_class & (iou >= match_iou), iou, torch.full_like(iou, -1))
        # Greedy highest-IoU matching. argmax's first-index tie break makes this deterministic.
        for _ in range(min(pidx.numel(), gidx.numel())):
            best, flat_index = scores.flatten().max(dim=0)
            if best < 0:
                break
            pi = torch.div(flat_index, scores.shape[1], rounding_mode="floor")
            gi = flat_index % scores.shape[1]
            pseudo_index, gt_index = pidx[pi], gidx[gi]
            pbox, gbox = pseudo_boxes[batch_idx, pseudo_index], gt_boxes[batch_idx, gt_index]
            denom = torch.stack((gbox[2] - gbox[0], gbox[3] - gbox[1], gbox[2] - gbox[0], gbox[3] - gbox[1])).clamp_min(1.0)
            selected[batch_idx, pseudo_index] = (pbox - gbox).abs() / denom >= error_threshold
            matched[batch_idx, pseudo_index] = True
            scores[pi, :] = -1
            scores[:, gi] = -1
    return selected, matched
