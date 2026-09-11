"""
Diagnostic-only instrumentation for investigating whether the transient ssod/cls_loss spikes
observed at very low labeled ratios (e.g. 1% CrowdHuman) actually harm training, or are harmless
optimization noise. See dfl_ssod_algorithm_flexible.md's sibling plan for the full protocol.

Nothing here changes normal training when `spike_diag` is not passed / disabled: SSODTrainer only
calls into this module from inside an `if self.spike_diag_enabled:` guard, so the design-freeze
requirement (no change to thresholds/schedule/etc. for ordinary runs) holds by construction.

Two things live here:
  1. JSONL per-iteration logging of loss/gradient/update statistics (`log_iteration_stats`).
  2. Full trainer-state snapshot/restore (model, teacher EMA, optimizer, GradScaler, RNG, and the
     exact labeled/unlabeled batches) so a spike iteration can be re-run under different
     interventions (NORMAL / SKIP-U / CLIP-U) from an identical starting point.
"""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _tensor_grad_norm(params) -> float:
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += p.grad.detach().float().pow(2).sum().item()
    return total**0.5


def compute_split_gradients(model: torch.nn.Module, loss_s: torch.Tensor, loss_u_scaled: torch.Tensor):
    """
    Compute ||g_s||, ||g_u|| and cos(g_s, g_u) via two separate `torch.autograd.grad` calls that
    do NOT populate `.grad` (so the real optimizer step is unaffected). `loss_u_scaled` should
    already include `ssod_weight` so the reported ||g_u|| matches what actually enters the
    combined update. Both losses must share the same graph (i.e. be produced in the same forward
    pass sequence as the real training step, called before the real combined `.backward()`).

    Returns:
        (norm_s, norm_u, cos_su, r_t): floats. r_t = norm_u / max(norm_s, eps).
    """
    params = [p for p in model.parameters() if p.requires_grad]
    g_s = torch.autograd.grad(loss_s, params, retain_graph=True, allow_unused=True)
    g_u = torch.autograd.grad(loss_u_scaled, params, retain_graph=True, allow_unused=True)

    flat_s = torch.cat([g.reshape(-1) for g, p in zip(g_s, params) if g is not None])
    flat_u = torch.cat([g.reshape(-1) for g, p in zip(g_u, params) if g is not None])

    norm_s = flat_s.norm().item()
    norm_u = flat_u.norm().item()
    denom = max(norm_s * norm_u, 1e-12)
    cos_su = torch.dot(flat_s, flat_u).item() / denom
    r_t = norm_u / max(norm_s, 1e-12)
    return norm_s, norm_u, cos_su, r_t


class SpikeCaptured(Exception):
    """Raised (only when spike_diag_stop_after_capture=True) right after a snapshot is saved, to
    force an immediate, unambiguous stop regardless of the surrounding training loop's own
    end-of-epoch stop/break bookkeeping. Callers should catch this around `model.train(...)`."""


def log_iteration_stats(log_path: str, record: dict[str, Any]) -> None:
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps(record) + "\n")


def rng_state_dict() -> dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def load_rng_state_dict(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def snapshot_trainer_state(trainer, labeled_batch: dict, unlabeled_batch: dict, epoch: int, ni: int) -> dict[str, Any]:
    """Capture everything needed to re-run one iteration identically under different interventions."""
    return {
        "epoch": epoch,
        "ni": ni,
        "model_state": copy.deepcopy(trainer.model.state_dict()),
        "teacher_state": copy.deepcopy(trainer.teacher.ema.state_dict()),
        "teacher_updates": trainer.teacher.updates,
        "optimizer_state": copy.deepcopy(trainer.optimizer.state_dict()),
        "scaler_state": copy.deepcopy(trainer.scaler.state_dict()),
        "rng_state": rng_state_dict(),
        # Detach + clone every tensor in the batch dicts so later in-place ops elsewhere can't corrupt them.
        "labeled_batch": {k: (v.detach().clone() if torch.is_tensor(v) else copy.deepcopy(v)) for k, v in labeled_batch.items()},
        "unlabeled_batch": {k: (v.detach().clone() if torch.is_tensor(v) else copy.deepcopy(v)) for k, v in unlabeled_batch.items()},
    }


def save_snapshot(snapshot: dict[str, Any], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(snapshot, path)


def load_snapshot(path: str) -> dict[str, Any]:
    return torch.load(path, weights_only=False)


def restore_trainer_state(trainer, snapshot: dict[str, Any]) -> None:
    trainer.model.load_state_dict(snapshot["model_state"])
    trainer.teacher.ema.load_state_dict(snapshot["teacher_state"])
    trainer.teacher.updates = snapshot["teacher_updates"]
    trainer.optimizer.load_state_dict(snapshot["optimizer_state"])
    trainer.scaler.load_state_dict(snapshot["scaler_state"])
    load_rng_state_dict(snapshot["rng_state"])


def apply_intervention(intervention: str, loss_s: torch.Tensor, loss_u: torch.Tensor, ssod_weight: float, g_ref: float | None, norm_u: float | None):
    """
    Combine supervised + unsupervised loss for ONE iteration under a named intervention.

    - normal:  g = g_s + ssod_weight * g_u                     (unchanged production behavior)
    - skip_u:  g = g_s                                          (drop the unsupervised update entirely)
    - clip_u:  g = g_s + ssod_weight * g_u * min(1, g_ref/||g_u||)   (keep direction, cap magnitude)

    clip_u requires the caller to have already measured norm_u (e.g. via compute_split_gradients)
    and to supply a pre-declared g_ref (not tuned post-hoc on this experiment's own results).
    """
    if intervention == "normal":
        return loss_s + ssod_weight * loss_u
    if intervention == "skip_u":
        return loss_s
    if intervention == "clip_u":
        assert g_ref is not None and norm_u is not None
        scale = min(1.0, g_ref / max(norm_u, 1e-12))
        return loss_s + ssod_weight * loss_u * scale
    raise ValueError(f"Unknown intervention: {intervention}")
