from __future__ import annotations

import torch


def localization_confidence(pred_distri: torch.Tensor, reg_max: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute DFL-based localization confidence for each predicted box.

    For every one of the 4 box edges, the DFL head predicts a discrete distribution over
    `reg_max` bins. A distribution concentrated on a single bin (or two adjacent bins)
    means the edge location is confidently regressed; a flat/multi-modal distribution
    means it is uncertain. Edge confidence is the probability mass on the argmax bin
    plus its strongest neighbor; box confidence is the minimum (i.e. weakest edge) over
    the 4 edges, since one badly localized edge makes the whole box unreliable.

    Args:
        pred_distri (torch.Tensor): Raw DFL logits, shape (..., 4, reg_max).
        reg_max (int): Number of DFL bins per edge.

    Returns:
        edge_conf (torch.Tensor): Per-edge confidence, shape (..., 4).
        box_conf (torch.Tensor): Per-box confidence (min over edges), shape (...).
    """
    p = pred_distri.softmax(-1)  # (..., 4, reg_max)
    p_max, k_max = p.max(-1)  # (..., 4)

    left_idx = (k_max - 1).clamp(min=0)
    right_idx = (k_max + 1).clamp(max=reg_max - 1)
    left_prob = p.gather(-1, left_idx.unsqueeze(-1)).squeeze(-1)
    right_prob = p.gather(-1, right_idx.unsqueeze(-1)).squeeze(-1)
    left_prob = torch.where(k_max > 0, left_prob, torch.zeros_like(left_prob))
    right_prob = torch.where(k_max < reg_max - 1, right_prob, torch.zeros_like(right_prob))

    edge_conf = p_max + torch.maximum(left_prob, right_prob)  # (..., 4)
    box_conf = edge_conf.amin(-1)  # (...)
    return edge_conf, box_conf
