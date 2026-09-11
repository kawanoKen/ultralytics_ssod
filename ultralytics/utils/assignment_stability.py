"""Deterministic DFL-supported perturbations and assignment-stability metrics for SSOD."""

from __future__ import annotations

import torch

from ultralytics.utils.tal import dist2bbox


PERTURBATION_NAMES = (
    "expectation",
    "left_q10",
    "left_q90",
    "top_q10",
    "top_q90",
    "right_q10",
    "right_q90",
    "bottom_q10",
    "bottom_q90",
)


def discrete_quantile(probabilities: torch.Tensor, quantile: float) -> torch.Tensor:
    """Return the first DFL bin whose CDF reaches ``quantile`` (no random sampling)."""
    if not 0.0 <= quantile <= 1.0:
        raise ValueError(f"quantile must be in [0, 1], got {quantile}")
    cdf = probabilities.cumsum(-1)
    threshold = torch.full_like(cdf[..., :1], quantile)
    return (cdf < threshold).sum(-1).to(probabilities.dtype)


def dfl_distribution_statistics(pred_distri: torch.Tensor) -> dict[str, torch.Tensor]:
    """Compute expectation, Q10/Q90, entropy, variance and deterministic width for (..., 4, R) logits."""
    probabilities = pred_distri.softmax(-1)
    bins = torch.arange(pred_distri.shape[-1], device=pred_distri.device, dtype=probabilities.dtype)
    expectation = (probabilities * bins).sum(-1)
    q10 = discrete_quantile(probabilities, 0.10)
    q90 = discrete_quantile(probabilities, 0.90)
    variance = (probabilities * (bins - expectation.unsqueeze(-1)).square()).sum(-1)
    entropy = -(probabilities * probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).log()).sum(-1)
    return {
        "probabilities": probabilities,
        "expectation": expectation,
        "q10": q10,
        "q90": q90,
        "width": q90 - q10,
        "variance": variance,
        "entropy": entropy,
    }


def dfl_supported_box_candidates(
    pred_distri: torch.Tensor,
    anchor_points: torch.Tensor,
    stride_tensor: torch.Tensor,
) -> torch.Tensor:
    """Decode expectation plus eight one-edge Q10/Q90 perturbations.

    Args:
        pred_distri: Raw DFL logits with shape (N, 4, R), ordered left/top/right/bottom.
        anchor_points: Grid points with shape (N, 2), in feature-grid coordinates.
        stride_tensor: Per-grid stride with shape (N, 1), or a broadcast-compatible equivalent.

    Returns:
        Pixel-coordinate XYXY boxes with shape (9, N, 4). Candidate zero is the expectation box.
    """
    if pred_distri.ndim != 3 or pred_distri.shape[-2] != 4:
        raise ValueError(f"pred_distri must have shape (N, 4, R), got {tuple(pred_distri.shape)}")
    stats = dfl_distribution_statistics(pred_distri)
    expectation, q10, q90 = stats["expectation"], stats["q10"], stats["q90"]
    distances = expectation.unsqueeze(0).repeat(len(PERTURBATION_NAMES), 1, 1)
    for edge in range(4):
        distances[1 + edge * 2, :, edge] = q10[:, edge]
        distances[2 + edge * 2, :, edge] = q90[:, edge]
    return dist2bbox(distances, anchor_points.unsqueeze(0), xywh=False) * stride_tensor.unsqueeze(0)


def geometric_box_candidates(
    baseline_boxes: torch.Tensor,
    mode: str,
    dfl_width: torch.Tensor | None = None,
    dfl_expectation: torch.Tensor | None = None,
    anchor_points: torch.Tensor | None = None,
    stride_tensor: torch.Tensor | None = None,
    reg_max: int | None = None,
    fixed_ratio: float = 0.05,
) -> torch.Tensor:
    """Build the fixed or DFL-width-matched M=9 controls used by analysis and training.

    Fixed uses 5% of box width for x edges and 5% of box height for y edges. Width-matched
    uses half the per-edge Q90-Q10 width symmetrically around the DFL expectation distance,
    clipped to the valid DFL distance range. Candidate zero is always ``baseline_boxes``.
    """
    candidates = baseline_boxes.unsqueeze(0).repeat(len(PERTURBATION_NAMES), 1, 1)
    if mode == "fixed":
        widths = baseline_boxes[:, 2] - baseline_boxes[:, 0]
        heights = baseline_boxes[:, 3] - baseline_boxes[:, 1]
        delta = torch.stack((widths, heights, widths, heights), -1) * fixed_ratio
        for edge in range(4):
            candidates[1 + edge * 2, :, edge] -= delta[:, edge]
            candidates[2 + edge * 2, :, edge] += delta[:, edge]
        return candidates
    if mode == "width_matched":
        required = (dfl_width, dfl_expectation, anchor_points, stride_tensor, reg_max)
        if any(value is None for value in required):
            raise ValueError("width_matched requires DFL width/expectation, anchors, stride, and reg_max")
        distances = dfl_expectation.unsqueeze(0).repeat(len(PERTURBATION_NAMES), 1, 1)
        delta = dfl_width * 0.5
        for edge in range(4):
            distances[1 + edge * 2, :, edge] -= delta[:, edge]
            distances[2 + edge * 2, :, edge] += delta[:, edge]
        distances.clamp_(0, reg_max - 1)
        candidates = dist2bbox(distances, anchor_points.unsqueeze(0), xywh=False) * stride_tensor.unsqueeze(0)
        candidates[0] = baseline_boxes
        return candidates
    raise ValueError(f"geometric mode must be fixed or width_matched, got {mode!r}")


def assignment_stability(
    assignment_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Summarize assignment identity over deterministic candidates.

    ``assignment_ids`` has shape (M, B, A), contains local object indices, and uses -1 for
    background. Candidate zero is the baseline expectation assignment.

    Returns:
        same_object_stability: Fraction of all M candidates retaining the baseline object.
        ambiguous_negative: Baseline background assigned to any object under a perturbation.
        identity_switch: Baseline foreground assigned to a different object at least once.
        foreground_frequency: Fraction of candidates for which each prediction is foreground.
    """
    if assignment_ids.ndim != 3 or assignment_ids.shape[0] < 1:
        raise ValueError(f"assignment_ids must have shape (M, B, A), got {tuple(assignment_ids.shape)}")
    baseline = assignment_ids[0]
    baseline_fg = baseline >= 0
    same_object = (assignment_ids == baseline.unsqueeze(0)) & baseline_fg.unsqueeze(0)
    same_object_stability = same_object.to(torch.float32).mean(0)
    foreground = assignment_ids >= 0
    ambiguous_negative = ~baseline_fg & foreground[1:].any(0) if assignment_ids.shape[0] > 1 else ~baseline_fg & False
    identity_switch = baseline_fg & ((assignment_ids >= 0) & (assignment_ids != baseline.unsqueeze(0))).any(0)
    foreground_frequency = foreground.to(torch.float32).mean(0)
    return same_object_stability, ambiguous_negative, identity_switch, foreground_frequency


def object_assignment_metrics(assignment_ids: torch.Tensor, num_objects: int) -> dict[str, torch.Tensor]:
    """Compute per-object Jaccard, instability, ambiguous ratio, and identity-switch counts."""
    if num_objects == 0:
        empty = assignment_ids.new_zeros((0,), dtype=torch.float32)
        return {"jaccard": empty, "instability": empty, "ambiguous_ratio": empty, "identity_switches": empty}
    baseline = assignment_ids[0]
    jaccards, ambiguous_ratios, switches = [], [], []
    for object_idx in range(num_objects):
        base = baseline == object_idx
        candidate_sets = assignment_ids[1:] == object_idx
        if candidate_sets.shape[0]:
            intersection = (candidate_sets & base).sum(-1).to(torch.float32)
            union = (candidate_sets | base).sum(-1).clamp_min(1).to(torch.float32)
            jaccard = (intersection / union).mean()
        else:
            jaccard = torch.ones((), device=assignment_ids.device)
        ever = (assignment_ids == object_idx).any(0)
        always = (assignment_ids == object_idx).all(0)
        ambiguous_ratios.append(((ever & ~always).sum() / ever.sum().clamp_min(1)).to(torch.float32))
        switched = ((assignment_ids[1:] >= 0) & (assignment_ids[1:] != object_idx)).any(0)
        switches.append(((baseline == object_idx) & switched).sum())
        jaccards.append(jaccard)
    jaccard = torch.stack(jaccards)
    return {
        "jaccard": jaccard,
        "instability": 1.0 - jaccard,
        "ambiguous_ratio": torch.stack(ambiguous_ratios),
        "identity_switches": torch.stack(switches).to(torch.float32),
    }
