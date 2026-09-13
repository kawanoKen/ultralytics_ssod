"""Short CPU/GPU smoke test for Oracle/DFL edge-wise DFL dose-response primitives.

This is intentionally not a training run. It executes selector, normalization,
DFL reduction, and backward for the six requested selector/weight combinations.
"""

from __future__ import annotations

import argparse

import torch

from ultralytics.utils.edge_dfl_reweight import (
    dfl_per_edge_loss,
    make_edge_dfl_weights,
    oracle_selected_edges,
    reduce_weighted_dfl,
    select_dfl_low_confidence_edges,
)


def oracle_mask(device: torch.device) -> torch.Tensor:
    pseudo = torch.tensor([[[1, 1, 9, 10]]], dtype=torch.float32, device=device)
    gt = torch.tensor([[[1, 1, 9, 8]]], dtype=torch.float32, device=device)
    cls = torch.zeros((1, 1, 1), device=device)
    valid = torch.ones((1, 1, 1), dtype=torch.bool, device=device)
    selected, _ = oracle_selected_edges(pseudo, cls, valid, gt, cls, valid, 10, 10, 0.10)
    return selected.reshape(1, 4)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    target = torch.tensor([[1.2, 2.1, 3.3, 4.4]], device=device)
    for selector in ("oracle", "dfl"):
        for weight in (0.0, 1.0, 20.0):
            logits = torch.randn((1, 4, 16), device=device, requires_grad=True)
            selected = oracle_mask(device) if selector == "oracle" else select_dfl_low_confidence_edges(
                torch.tensor([[0.9, 0.9, 0.9, 0.2]], device=device), 0.6
            )
            weights = make_edge_dfl_weights(selected, weight, normalize=True)
            loss = reduce_weighted_dfl(dfl_per_edge_loss(logits, target, 16), torch.ones((1, 1), device=device), 1.0, weights.normalized)
            loss.backward()
            if not torch.isfinite(loss) or not torch.isfinite(logits.grad).all():
                raise RuntimeError(f"non-finite result for selector={selector} weight={weight}")
            print(
                f"selector={selector} weight={weight:g} loss={loss.item():.6f} "
                f"selected={int(selected.sum())}/{selected.numel()} mean_weight={weights.normalized.mean().item():.6f}"
            )


if __name__ == "__main__":
    main()
