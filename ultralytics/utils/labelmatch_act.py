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
        max_det: int = 300,
        reliable_ratio: float = 0.2,
        log_dir: str | Path | None = None,
    ):
        if mode not in ACT_MODES:
            raise ValueError(f"labelmatch_mode must be one of {sorted(ACT_MODES)}, got {mode!r}")
        if update_interval < 1:
            raise ValueError("labelmatch_update_interval must be >= 1")
        if not 0.0 <= candidate_conf_floor < 1.0:
            raise ValueError("labelmatch_candidate_conf_floor must be in [0, 1)")
        if not 0.0 < reliable_ratio <= 1.0:
            raise ValueError("labelmatch_reliable_ratio must be in (0, 1]")
        self.rho = np.asarray(rho, dtype=np.float64)
        self.nc = len(self.rho)
        self.mode = mode
        self.update_interval = update_interval
        self.candidate_conf_floor = candidate_conf_floor
        self.nms_iou = nms_iou
        self.max_det = max_det
        self.reliable_ratio = reliable_ratio
        # t_c: K_c-th score (candidate); t_c^r: round(alpha*K_c)-th score (reliable). [t_c, t_c^r) is ignored.
        self.candidate_thresholds = torch.ones(self.nc)
        self.reliable_thresholds = torch.ones(self.nc)
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
            target_reliable = int(round(self.reliable_ratio * target))
            scores = per_class_scores[c]
            t_c = act_threshold(scores, target, self.candidate_conf_floor)
            t_r = max(act_threshold(scores, target_reliable, self.candidate_conf_floor), t_c)
            self.candidate_thresholds[c] = t_c
            self.reliable_thresholds[c] = t_r
            stats.append(
                {
                    "iteration": iteration,
                    "epoch": epoch,
                    "class": c,
                    "candidate_threshold": t_c,
                    "reliable_threshold": t_r,
                    "reliable_ratio": self.reliable_ratio,
                    "rho_L_labeled_boxes_per_image": float(self.rho[c]),
                    "probe_images": num_images,
                    "target_candidate_count": target,
                    "target_reliable_count": target_reliable,
                    "num_candidates_postnms": int(per_class_candidates[c]),
                    "target_reachable": int(per_class_candidates[c]) >= target,
                    "probe_seconds": seconds,
                    "_candidates": scores[scores >= t_c] if scores.numel() else scores,
                    "_reliable": scores[scores >= t_r] if scores.numel() else scores,
                }
            )
        self.last_update_iter = iteration
        self.last_stats = stats
        return self.candidate_thresholds, self.reliable_thresholds

    @torch.no_grad()
    def run_offline_probe(self, teacher, loader, device, iteration, epoch, amp=False):
        """Teacher inference -> floor-only NMS -> gather top-K per class across ranks -> thresholds."""
        import torchvision  # noqa: F401  (non_max_suppression uses torchvision.ops.nms only once imported)

        start = time.perf_counter()
        ddp = dist.is_available() and dist.is_initialized()
        local_scores = [[] for _ in range(self.nc)]
        local_images = local_prenms = 0
        saturated_min_scores = []  # lowest kept score of images whose NMS output hit max_det
        for batch in loader:
            img = batch["img"].to(device, non_blocking=True).float() / 255
            with autocast(bool(amp), device.type):
                out = teacher(img)
            preds = (out[0] if isinstance(out, (tuple, list)) else out).float()
            local_prenms += int((preds[:, 4 : 4 + self.nc].amax(1) > self.candidate_conf_floor).sum())
            dets = non_max_suppression(
                preds,
                conf_thres=self.candidate_conf_floor,
                iou_thres=self.nms_iou,
                max_det=self.max_det,
                max_time_img=ACT_NMS_MAX_TIME_IMG,
            )
            local_images += img.shape[0]
            for det in dets:
                if det.shape[0] >= self.max_det:
                    saturated_min_scores.append(float(det[:, 4].min()))
                if det.numel():
                    for c in range(self.nc):
                        local_scores[c].append(det[det[:, 5] == c, 4].float().cpu())
        local_scores = [torch.cat(s) if s else torch.empty(0) for s in local_scores]
        counts = torch.tensor(
            [s.numel() for s in local_scores]
            + [float(s.sum()) for s in local_scores]
            + [local_images, local_prenms, len(saturated_min_scores)],
            dtype=torch.float64,
            device=device,
        )
        if ddp:
            dist.all_reduce(counts)
        num_images = int(counts[-3].item())
        targets = [int(round(self.rho[c] * num_images)) for c in range(self.nc)]
        # The global top-K lies inside the union of per-rank top-Ks, so only those are gathered.
        local_top = [s.topk(min(max(targets[c], 0), s.numel())).values for c, s in enumerate(local_scores)]
        if ddp:
            gathered = [None] * dist.get_world_size()
            dist.all_gather_object(gathered, [t.numpy() for t in local_top])
            merged = [torch.from_numpy(np.concatenate([g[c] for g in gathered])) for c in range(self.nc)]
        else:
            merged = local_top
        per_class_candidates = counts[: self.nc].tolist()
        self.update_from_scores(merged, per_class_candidates, num_images, iteration, epoch, time.perf_counter() - start)
        # Count every box at or above each threshold (can exceed the target only through ties), and
        # images where max_det truncated boxes that would still clear t_c (the cap then binds on ACT).
        min_t_c = float(self.candidate_thresholds.min())
        post = torch.tensor(
            [float((local_scores[c] >= self.candidate_thresholds[c]).sum()) for c in range(self.nc)]
            + [float((local_scores[c] >= self.reliable_thresholds[c]).sum()) for c in range(self.nc)]
            + [float(sum(s >= min_t_c for s in saturated_min_scores))],
            dtype=torch.float64,
            device=device,
        )
        if ddp:
            dist.all_reduce(post)
        m = max(num_images, 1)
        for c, row in enumerate(self.last_stats):
            n_post = per_class_candidates[c]
            n_cand, n_rel = int(post[c].item()), int(post[self.nc + c].item())
            row["candidate_count_probe"] = n_cand
            row["reliable_count_probe"] = n_rel
            row["uncertain_count_probe"] = n_cand - n_rel
            row["candidate_boxes_per_image"] = n_cand / m
            row["reliable_boxes_per_image"] = n_rel / m
            row["uncertain_boxes_per_image"] = (n_cand - n_rel) / m
            row["reliable_fraction_of_candidates"] = n_rel / n_cand if n_cand else float("nan")
            row["conf_mean_all_postnms"] = float(counts[self.nc + c].item()) / n_post if n_post else float("nan")
            row["num_candidates_prenms"] = int(counts[-2].item())
            row["images_at_max_det"] = int(counts[-1].item())
            row["images_max_det_binding"] = int(post[-1].item())
        return self.candidate_thresholds, self.reliable_thresholds

    def log_last_update(self) -> None:
        if self._csv is None:
            return
        for row in self.last_stats:
            out = {k: v for k, v in row.items() if not k.startswith("_")}
            for name in ("candidates", "reliable"):
                scores = row[f"_{name}"]
                out[f"conf_mean_{name}"] = float(scores.mean()) if scores.numel() else float("nan")
                out[f"conf_median_{name}"] = float(scores.median()) if scores.numel() else float("nan")
            self._csv.write(out)

    def close(self) -> None:
        if self._csv is not None:
            self._csv.close()
