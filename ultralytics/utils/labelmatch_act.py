"""LabelMatch Adaptive Confidence Threshold (ACT) for SSOD pseudo-label selection.

Implements Eq. (5) of LabelMatch (Chen et al., CVPR 2022): the class-wise threshold is the
K_c-th highest post-NMS teacher score on an unlabeled probe subset, where
K_c = round(labeled GT boxes per image of class c * number of probe images).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from torch import distributed as dist

from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.ssod_diagnostics import _CsvWriter
from ultralytics.utils.torch_utils import autocast

ACT_MODES = {"offline_probe"}
# non_max_suppression silently leaves the rest of a batch empty once its time limit trips; with a
# 0.001 floor on crowded scenes that would drop candidates, so ACT effectively disables the limit.
ACT_NMS_MAX_TIME_IMG = 10.0


def labeled_boxes_per_image(labels: list[dict], nc: int) -> np.ndarray:
    """Return rho_c = (#labeled GT boxes of class c) / (#labeled images), images without boxes included."""
    counts = np.zeros(nc, dtype=np.float64)
    for label in labels:
        cls = np.asarray(label.get("cls", []), dtype=np.int64).reshape(-1)
        if cls.size:
            counts += np.bincount(cls, minlength=nc)[:nc]
    return counts / max(len(labels), 1)


def act_threshold(scores: torch.Tensor, target_count: int, floor: float) -> float:
    """Return the ``target_count``-th highest finite score (1-indexed), with LabelMatch boundary rules."""
    if target_count <= 0:
        return 1.0
    scores = scores[torch.isfinite(scores)]
    if scores.numel() < target_count:
        return float(floor)  # too few candidates: admit every candidate that cleared the floor
    return float(torch.topk(scores, target_count).values[-1])


class ACTManager:
    """Holds class-wise ACT thresholds and refreshes them from teacher scores on unlabeled data.

    ``update_from_scores`` is the mode-independent core; ``run_offline_probe`` is the offline
    score source. An online score queue can later feed the same core without touching training.
    """

    def __init__(
        self,
        rho: np.ndarray,
        mode: str = "offline_probe",
        update_interval: int = 1000,
        candidate_conf_floor: float = 0.001,
        nms_iou: float = 0.65,
        log_dir: str | Path | None = None,
    ):
        if mode not in ACT_MODES:
            raise ValueError(f"labelmatch_mode must be one of {sorted(ACT_MODES)}, got {mode!r}")
        if update_interval < 1:
            raise ValueError("labelmatch_update_interval must be >= 1")
        if not 0.0 <= candidate_conf_floor < 1.0:
            raise ValueError("labelmatch_candidate_conf_floor must be in [0, 1)")
        self.rho = np.asarray(rho, dtype=np.float64)
        self.nc = len(self.rho)
        self.mode = mode
        self.update_interval = update_interval
        self.candidate_conf_floor = candidate_conf_floor
        self.nms_iou = nms_iou
        self.thresholds = torch.ones(self.nc)
        self.last_update_iter = None
        self.last_stats: list[dict] = []
        self._csv = _CsvWriter(Path(log_dir) / "labelmatch_act.csv") if log_dir is not None else None

    def should_update(self, iteration: int) -> bool:
        return self.last_update_iter is None or iteration - self.last_update_iter >= self.update_interval

    def update_from_scores(self, per_class_scores, per_class_candidates, num_images, iteration, epoch, seconds=0.0):
        """Set thresholds from gathered per-class scores (already restricted to each class's top K_c)."""
        stats = []
        for c in range(self.nc):
            target = int(round(self.rho[c] * num_images))
            scores = per_class_scores[c]
            tau = act_threshold(scores, target, self.candidate_conf_floor)
            selected = scores[scores >= tau] if scores.numel() else scores
            self.thresholds[c] = tau
            stats.append(
                {
                    "iteration": iteration,
                    "epoch": epoch,
                    "class": c,
                    "threshold": tau,
                    "labeled_boxes_per_image": float(self.rho[c]),
                    "probe_images": num_images,
                    "target_pseudo_count": target,
                    "num_candidates": int(per_class_candidates[c]),
                    "probe_seconds": seconds,
                    "_selected": selected,
                }
            )
        self.last_update_iter = iteration
        self.last_stats = stats
        return self.thresholds

    @torch.no_grad()
    def run_offline_probe(self, teacher, loader, device, iteration, epoch, amp=False):
        """Teacher inference -> floor-only NMS -> gather top-K per class across ranks -> thresholds."""
        import torchvision  # noqa: F401  (non_max_suppression uses torchvision.ops.nms only once imported)

        start = time.perf_counter()
        ddp = dist.is_available() and dist.is_initialized()
        local_scores = [[] for _ in range(self.nc)]
        local_images = 0
        for batch in loader:
            img = batch["img"].to(device, non_blocking=True).float() / 255
            with autocast(bool(amp), device.type):
                out = teacher(img)
            preds = out[0] if isinstance(out, (tuple, list)) else out
            dets = non_max_suppression(
                preds.float(),
                conf_thres=self.candidate_conf_floor,
                iou_thres=self.nms_iou,
                max_time_img=ACT_NMS_MAX_TIME_IMG,
            )
            local_images += img.shape[0]
            for det in dets:
                if det.numel():
                    for c in range(self.nc):
                        local_scores[c].append(det[det[:, 5] == c, 4].float().cpu())
        local_scores = [torch.cat(s) if s else torch.empty(0) for s in local_scores]
        counts = torch.tensor([s.numel() for s in local_scores] + [local_images], dtype=torch.float64, device=device)
        if ddp:
            dist.all_reduce(counts)
        num_images = int(counts[-1].item())
        targets = [int(round(self.rho[c] * num_images)) for c in range(self.nc)]
        # The global top-K lies inside the union of per-rank top-Ks, so only those are gathered.
        local_top = [s.topk(min(max(targets[c], 0), s.numel())).values for c, s in enumerate(local_scores)]
        if ddp:
            gathered = [None] * dist.get_world_size()
            dist.all_gather_object(gathered, [t.numpy() for t in local_top])
            merged = [torch.from_numpy(np.concatenate([g[c] for g in gathered])) for c in range(self.nc)]
        else:
            merged = local_top
        self.update_from_scores(merged, counts[:-1].tolist(), num_images, iteration, epoch, time.perf_counter() - start)
        # Count every candidate at or above the new threshold (can exceed K_c only through ties).
        actual = torch.tensor(
            [float((local_scores[c] >= self.thresholds[c]).sum()) for c in range(self.nc)],
            dtype=torch.float64,
            device=device,
        )
        if ddp:
            dist.all_reduce(actual)
        for c, row in enumerate(self.last_stats):
            row["actual_pseudo_count"] = int(actual[c].item())
            row["pseudo_boxes_per_image"] = row["actual_pseudo_count"] / max(num_images, 1)
        return self.thresholds

    def log_last_update(self) -> None:
        if self._csv is None:
            return
        for row in self.last_stats:
            selected = row["_selected"]
            out = {k: v for k, v in row.items() if not k.startswith("_")}
            out["conf_mean_selected"] = float(selected.mean()) if selected.numel() else float("nan")
            out["conf_median_selected"] = float(selected.median()) if selected.numel() else float("nan")
            self._csv.write(out)

    def close(self) -> None:
        if self._csv is not None:
            self._csv.close()
