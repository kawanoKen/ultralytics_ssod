# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Training-time-only diagnostic logging for SSOD.

Everything logged here is only observable during the live training forward pass
(EMA update counters, per-step fg_mask / assignment counts, teacher pre-NMS candidate
counts, DFL entropy of pseudo-label candidates before they are filtered, ...). None
of it is meant to be recomputable later from a checkpoint or from running inference
on saved predictions -- that data belongs in the validator / plotting scripts instead.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ultralytics.utils.metrics import bbox_iou, box_iou
from ultralytics.utils.tal import bbox2dist, make_anchors


@torch.no_grad()
def per_anchor_bbox_terms(bbox_loss, pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, fg_mask):
    """Unnormalized per-foreground-anchor box/DFL terms, exactly as BboxLoss.forward sums them."""
    weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
    iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
    box_term = ((1.0 - iou) * weight).view(-1)
    if bbox_loss.dfl_loss:
        reg_max = bbox_loss.dfl_loss.reg_max
        target_ltrb = bbox2dist(anchor_points, target_bboxes, reg_max - 1)
        dfl_term = (bbox_loss.dfl_loss(pred_distri[fg_mask].view(-1, reg_max), target_ltrb[fg_mask]) * weight).view(-1)
    else:
        dfl_term = torch.zeros_like(box_term)
    return box_term, dfl_term


def _per_image_sum(values, fg_mask):
    """Scatter per-foreground-anchor values back to their image index -> (batch,)."""
    out = torch.zeros(fg_mask.shape[0], device=fg_mask.device, dtype=torch.float32)
    if values.numel():
        out.index_add_(0, fg_mask.nonzero(as_tuple=True)[0], values.float())
    return out


@torch.no_grad()
def compute_supervised_per_image_loss(criterion, preds, batch) -> dict:
    """Per-image decomposition of the supervised v8DetectionLoss (sums over images = loss_items).

    No extra model forward: ``preds`` are the raw predictions of the real labeled forward.
    """
    feats = preds[1] if isinstance(preds, tuple) else preds
    pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], criterion.no, -1) for xi in feats], 2).split(
        (criterion.reg_max * 4, criterion.nc), 1
    )
    pred_scores = pred_scores.permute(0, 2, 1).contiguous()
    pred_distri = pred_distri.permute(0, 2, 1).contiguous()
    dtype = pred_scores.dtype
    batch_size = pred_scores.shape[0]
    imgsz = torch.tensor(feats[0].shape[2:], device=criterion.device, dtype=dtype) * criterion.stride[0]
    anchor_points, stride_tensor = make_anchors(feats, criterion.stride, 0.5)
    targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1).to(criterion.device)
    targets = criterion.preprocess(targets, batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
    gt_labels, gt_bboxes = targets.split((1, 4), 2)
    mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
    pred_bboxes = criterion.bbox_decode(anchor_points, pred_distri)
    _, target_bboxes, target_scores, fg_mask, _ = criterion.assigner(
        pred_scores.detach().sigmoid(),
        (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
        anchor_points * stride_tensor,
        gt_labels,
        gt_bboxes,
        mask_gt,
    )
    fg_mask = fg_mask.bool()
    tss = max(float(target_scores.sum()), 1.0)
    cls = criterion.bce(pred_scores, target_scores.to(dtype)).float().sum(-1).sum(1) / tss * criterion.hyp.cls
    box = torch.zeros(batch_size, device=fg_mask.device)
    dfl = torch.zeros(batch_size, device=fg_mask.device)
    if fg_mask.any():
        box_t, dfl_t = per_anchor_bbox_terms(
            criterion.bbox_loss, pred_distri, pred_bboxes, anchor_points, target_bboxes / stride_tensor, target_scores, fg_mask
        )
        box = _per_image_sum(box_t, fg_mask) / tss * criterion.hyp.box
        dfl = _per_image_sum(dfl_t, fg_mask) / tss * criterion.hyp.dfl
    return {
        "n_gt": mask_gt.squeeze(-1).sum(1),
        "pos_anchors": fg_mask.sum(1),
        "target_score_sum": target_scores.sum(-1).sum(1),
        "cls": cls,
        "box": box,
        "dfl": dfl,
    }


_STATUS_NAMES = {2: "reliable", 1: "ignored", 0: "discarded"}


def _to_bgr(img: torch.Tensor) -> np.ndarray:
    return (img.detach().float().clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()[:, :, ::-1].copy()


class EpochSampleLogger:
    """Per-sample records (every unlabeled and labeled image) for the first ``epochs`` SSOD epochs.

    Written by every DDP rank to its own files, so the union covers all samples:
      epoch_samples/unlabeled_rank{r}.jsonl : per unlabeled image -- all teacher pseudo boxes
          [x1,y1,x2,y2,conf,cls,status,max_IoU_to_GT] (status 2=reliable,1=ignored,0=discarded),
          transformed GT, per-image unsupervised loss contribution
      epoch_samples/labeled_rank{r}.jsonl   : per labeled image -- per-image supervised loss
      epoch_samples/steps_rank{r}.jsonl     : per step -- thresholds, denominators, weights, totals
      epoch_samples/vis/rank{r}/*.jpg       : weak (teacher) | strong (student) views with GT (blue),
          reliable (green), ignored (yellow); discarded boxes are only in the jsonl
    GT of unlabeled images is used for this offline record only, never for training.
    """

    def __init__(self, save_dir, rank: int, epochs: int = 1, visualize: bool = True):
        self.epochs = epochs
        self.rank = max(rank, 0)
        self.visualize = visualize
        self.dir = Path(save_dir) / "epoch_samples"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.vis_dir = self.dir / "vis" / f"rank{self.rank}"
        if visualize:
            self.vis_dir.mkdir(parents=True, exist_ok=True)
        self._files = {k: open(self.dir / f"{k}_rank{self.rank}.jsonl", "a", encoding="utf-8") for k in ("unlabeled", "labeled", "steps")}

    def active(self, epoch: int) -> bool:
        return epoch < self.epochs

    def _write(self, key, row):
        self._files[key].write(json.dumps(row) + "\n")

    def log_step(self, row: dict) -> None:
        self._write("steps", {"rank": self.rank, **row})
        for f in self._files.values():
            f.flush()

    def log_labeled(self, step, epoch, im_files, per_image: dict, loss_scale: float) -> None:
        n = len(im_files)
        vals = {k: v.detach().float().cpu().tolist() for k, v in per_image.items()}
        for b in range(n):
            loss_sum = vals["cls"][b] + vals["box"][b] + vals["dfl"][b]
            self._write(
                "labeled",
                {
                    "epoch": epoch, "step": step, "rank": self.rank, "batch_pos": b, "im_file": im_files[b],
                    **{k: vals[k][b] for k in vals},
                    "loss_sum": loss_sum, "effective_contribution": loss_sum * loss_scale,
                },
            )

    def log_unlabeled(self, step, epoch, im_files, img_weak, img_strong, pseudo, status, gt_xyxy, gt_bidx, per_image, loss_scale):
        """``pseudo``: (N,7) pixel x1,y1,x2,y2,conf,cls,batch_idx; ``status``: (N,) int; GT pixel xyxy + batch idx."""
        vals = {k: v.detach().float().cpu().tolist() for k, v in per_image.items()} if per_image else {}
        pseudo, status, gt_xyxy, gt_bidx = pseudo.float().cpu(), status.cpu(), gt_xyxy.float().cpu(), gt_bidx.cpu()
        for b in range(len(im_files)):
            pm = pseudo[:, 6] == b
            p, s, g = pseudo[pm], status[pm], gt_xyxy[gt_bidx == b]
            max_iou = box_iou(p[:, :4], g).max(1).values if len(p) and len(g) else torch.zeros(len(p))
            counts = {name: int((s == code).sum()) for code, name in _STATUS_NAMES.items()}
            tp = {name: int(((s == code) & (max_iou >= 0.5)).sum()) for code, name in _STATUS_NAMES.items()}
            loss = {k: vals[k][b] for k in vals}
            loss_sum = sum(loss.get(k, 0.0) or 0.0 for k in ("cls", "box", "dfl"))
            row = {
                "epoch": epoch, "step": step, "rank": self.rank, "batch_pos": b, "im_file": im_files[b],
                "n_gt": len(g), "n_postnms": len(p),
                **{f"n_{k}": v for k, v in counts.items()},
                **{f"tp50_{k}": v for k, v in tp.items()},
                "conf_max": float(p[:, 4].max()) if len(p) else None,
                "loss": {**loss, "loss_sum": loss_sum, "effective_contribution": loss_sum * loss_scale},
                # [x1, y1, x2, y2, conf, cls, status, max_IoU_to_GT]
                "boxes": [
                    [round(float(v), 2) for v in p[j, :4]]
                    + [round(float(p[j, 4]), 4), int(p[j, 5]), int(s[j]), round(float(max_iou[j]), 3)]
                    for j in range(len(p))
                ],
                "gt": [[round(float(v), 1) for v in r] for r in g],
            }
            self._write("unlabeled", row)
            if self.visualize:
                self._draw(step, epoch, b, im_files[b], img_weak[b], img_strong[b] if img_strong is not None else None, p, s, g, row)

    def _draw(self, step, epoch, b, im_file, weak, strong, p, s, g, row):
        import cv2

        panels = []
        for img in (weak, strong) if strong is not None else (weak,):
            canvas = _to_bgr(img)
            for x1, y1, x2, y2 in g.tolist():
                cv2.rectangle(canvas, (int(x1), int(y1)), (int(x2), int(y2)), (255, 128, 0), 1)
            for j in range(len(p)):
                if int(s[j]) == 0:
                    continue
                color, thick = ((0, 220, 0), 2) if int(s[j]) == 2 else ((0, 220, 255), 1)
                x1, y1, x2, y2 = (int(v) for v in p[j, :4].tolist())
                cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thick)
                if int(s[j]) == 2:
                    cv2.putText(canvas, f"{float(p[j, 4]):.2f}", (x1, max(y1 - 2, 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
            panels.append(canvas)
        img = np.concatenate(panels, 1)
        loss = row["loss"]
        header = np.zeros((36, img.shape[1], 3), dtype=np.uint8)
        lines = [
            f"ep{epoch} step{step} r{self.rank} {Path(im_file).stem}  GT {row['n_gt']}  reliable {row['n_reliable']} "
            f"(TP50 {row['tp50_reliable']})  ignored {row['n_ignored']} (TP50 {row['tp50_ignored']})  discarded {row['n_discarded']}",
            "  ".join(f"{k} {v:.4g}" for k, v in loss.items() if isinstance(v, (int, float)) and v is not None),
        ]
        for k, line in enumerate(lines):
            cv2.putText(header, line, (4, 14 + 16 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
        cv2.imwrite(str(self.vis_dir / f"e{epoch}_s{step:06d}_b{b:03d}_{Path(im_file).stem}.jpg"), np.concatenate((header, img), 0),
                    [cv2.IMWRITE_JPEG_QUALITY, 85])

    def close(self) -> None:
        for f in self._files.values():
            f.close()


class _CsvWriter:
    """Append-only CSV writer that infers its header from the first row it sees."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fieldnames = None
        self._file = None
        self._writer = None

    def write(self, row: dict) -> None:
        if self._writer is None:
            self._fieldnames = list(row.keys())
            new_file = not self.path.exists() or self.path.stat().st_size == 0
            self._file = self.path.open("a", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(self._file, fieldnames=self._fieldnames)
            if new_file:
                self._writer.writeheader()
        # A diagnostics logging bug should never be able to kill an hours-long training run:
        # silently drop keys the header doesn't have and fill any the row is missing, rather
        # than letting DictWriter raise on a schema mismatch.
        row = {k: row.get(k) for k in self._fieldnames}
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
            self._writer = None


def dfl_entropy(box_distri: torch.Tensor) -> torch.Tensor:
    """Mean (over the 4 edges) Shannon entropy of the DFL softmax distribution per box.

    Args:
        box_distri (torch.Tensor): (N, 4, reg_max) raw DFL logits for N candidate boxes.

    Returns:
        (torch.Tensor): (N,) entropy in nats, averaged over the 4 edges.
    """
    if box_distri.numel() == 0:
        return torch.zeros(0, device=box_distri.device)
    log_p = box_distri.log_softmax(-1)
    p = log_p.exp()
    return -(p * log_p).sum(-1).mean(-1)


@torch.no_grad()
def compute_supervised_assignment_stats(criterion, preds, batch) -> dict:
    """Redo just the TaskAlignedAssigner step for the labeled batch to expose the same
    positive-anchor / target-score statistics EfficientTeacherLoss already tracks for the
    unlabeled batch, so the two are directly comparable. No-op with respect to gradients
    (everything here is `no_grad`) and does not perform an extra model forward pass --
    `preds` must be the raw predictions already produced by the student's labeled forward.
    """
    feats = preds[1] if isinstance(preds, tuple) else preds
    pred_distri, pred_scores = torch.cat(
        [xi.view(feats[0].shape[0], criterion.no, -1) for xi in feats], 2
    ).split((criterion.reg_max * 4, criterion.nc), 1)
    pred_scores = pred_scores.permute(0, 2, 1).contiguous()
    pred_distri = pred_distri.permute(0, 2, 1).contiguous()
    dtype = pred_scores.dtype
    batch_size = pred_scores.shape[0]
    imgsz = torch.tensor(feats[0].shape[2:], device=criterion.device, dtype=dtype) * criterion.stride[0]
    anchor_points, stride_tensor = make_anchors(feats, criterion.stride, 0.5)

    targets = torch.cat(
        (batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1
    ).to(criterion.device)
    targets = criterion.preprocess(targets, batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
    gt_labels, gt_bboxes = targets.split((1, 4), 2)
    mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

    pred_bboxes = criterion.bbox_decode(anchor_points, pred_distri)
    _, _, target_scores, fg_mask, target_gt_idx = criterion.assigner(
        pred_scores.detach().sigmoid(),
        (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
        anchor_points * stride_tensor,
        gt_labels,
        gt_bboxes,
        mask_gt,
    )
    fg_mask = fg_mask.bool()
    # See the matching comment in EfficientTeacherLoss._compute_assignment_stats: the assigner's
    # n_max_boxes==0 fast path returns target_gt_idx in pred_scores' dtype, not long, and
    # preprocess() returns a zero-width padded dim when the batch has no GT boxes at all.
    target_gt_idx = target_gt_idx.long()
    n_max_boxes = mask_gt.shape[1]
    if n_max_boxes == 0:
        per_box_counts = torch.zeros(0, device=fg_mask.device)
    else:
        counts = torch.zeros(mask_gt.shape[0], n_max_boxes, device=fg_mask.device, dtype=torch.float32)
        idx = torch.where(fg_mask, target_gt_idx, torch.zeros_like(target_gt_idx))
        counts.scatter_add_(1, idx, fg_mask.to(counts.dtype))
        per_box_counts = counts[mask_gt.squeeze(-1).bool()]

    return {
        "num_sup_positive": int(fg_mask.sum()),
        "num_sup_assigned": int(mask_gt.sum()),
        "sum_target_scores_sup": float(target_scores.sum()),
        "assigned_anchors_per_sup_box_mean": float(per_box_counts.mean()) if per_box_counts.numel() else 0.0,
        "assigned_anchors_per_sup_box_std": float(per_box_counts.std()) if per_box_counts.numel() > 1 else 0.0,
        "target_score_mean_sup": float(target_scores.sum(-1)[fg_mask].mean()) if fg_mask.any() else 0.0,
        "target_score_std_sup": float(target_scores.sum(-1)[fg_mask].std()) if fg_mask.sum() > 1 else 0.0,
    }


def _quantiles(x: torch.Tensor, qs=(0.10, 0.25, 0.50, 0.75, 0.90)):
    if x.numel() == 0:
        return {f"q{int(q * 100)}": None for q in qs}
    q_vals = torch.quantile(x.float(), torch.tensor(qs, device=x.device))
    return {f"q{int(q * 100)}": float(v) for q, v in zip(qs, q_vals)}


class BNMismatchProbe:
    """Hooks every BatchNorm2d layer under the given name prefixes (e.g. "model.21", "model.22" --
    the Detect head and the neck block feeding it, identified in bn_mismatch_diagnosis2.py as
    where the teacher's stored running stats diverge most catastrophically from the live batch)
    and, on each forward pass, compares that layer's stored running_mean/running_var against the
    empirical mean/var of the batch that just went through it. Meant to be attached once to
    `self.teacher.ema` and read after every real pseudo-labeling forward call during training --
    this is a live, training-time measurement, not something recomputable from a checkpoint alone
    (the checkpoint has the running stats, but not what a specific live batch's activations were).
    """

    def __init__(self, model: nn.Module, name_prefixes: tuple[str, ...] = ("model.21", "model.22")):
        self._captured: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self._handles = []
        for name, m in model.named_modules():
            if isinstance(m, nn.BatchNorm2d) and any(name.startswith(p) for p in name_prefixes):
                self._handles.append(m.register_forward_hook(self._make_hook(name)))

    def _make_hook(self, name: str):
        def hook(module, inp, out):
            x = inp[0].detach()
            mean = x.mean(dim=(0, 2, 3))
            var = x.var(dim=(0, 2, 3), unbiased=False)
            self._captured[name] = (mean, var, module.running_mean.detach(), module.running_var.detach())

        return hook

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    def summarize(self) -> dict:
        """Aggregate mismatch across all hooked layers from the most recent forward pass, plus
        the single worst-offending layer by variance ratio."""
        if not self._captured:
            return {}
        worst_name, worst_ratio, worst_deviation = None, None, -1.0
        mean_mismatches, var_ratios = [], []
        for name, (emp_mean, emp_var, stored_mean, stored_var) in self._captured.items():
            mean_mismatches.append((emp_mean - stored_mean).abs().mean().item())
            ratio = (emp_var / stored_var.clamp_min(1e-12)).mean().item()
            var_ratios.append(ratio)
            deviation = abs(ratio - 1.0)
            if deviation > worst_deviation:
                worst_name, worst_ratio, worst_deviation = name, ratio, deviation
        return {
            "bn_mean_mismatch_avg": sum(mean_mismatches) / len(mean_mismatches),
            "bn_var_ratio_avg": sum(var_ratios) / len(var_ratios),
            "bn_worst_layer": worst_name,
            "bn_worst_layer_var_ratio": worst_ratio,
        }


class SSODDiagnosticsLogger:
    """Owns the four training-dynamics CSVs for one SSODTrainer run."""

    def __init__(self, save_dir, ema_interval: int = 100, dist_interval: int = 100):
        save_dir = Path(save_dir)
        self.ema_interval = ema_interval
        self.dist_interval = dist_interval
        self.ema_csv = _CsvWriter(save_dir / "logs" / "ema_diagnostics.csv")
        self.grad_norm_csv = _CsvWriter(save_dir / "logs" / "grad_norm_dynamics.csv")
        self.bn_mismatch_csv = _CsvWriter(save_dir / "logs" / "bn_mismatch_dynamics.csv")
        self.dynamics_csv = _CsvWriter(save_dir / "logs" / "training_dynamics.csv")
        self.pseudo_csv = _CsvWriter(save_dir / "logs" / "pseudo_label_dynamics.csv")
        self.assign_csv = _CsvWriter(save_dir / "logs" / "assignment_dynamics.csv")

    def close(self) -> None:
        for w in (self.ema_csv, self.dynamics_csv, self.pseudo_csv, self.assign_csv, self.grad_norm_csv, self.bn_mismatch_csv):
            w.close()

    def maybe_log_bn_mismatch(self, step: int, epoch: int, probe: "BNMismatchProbe") -> None:
        if step % self.dist_interval != 0:
            return
        summary = probe.summarize()
        if not summary:
            return
        self.bn_mismatch_csv.write({"step": step, "epoch": epoch, **summary})

    def log_grad_norm(
        self, step: int, epoch: int, grad_norm: float, max_norm: float = 10.0, total_loss: float | None = None
    ) -> None:
        """Every optimizer_step() call: the pre-clip global gradient norm clip_grad_norm_ already
        computes, and whether this step actually got clipped -- logged unconditionally (every
        step, not just at should_log_dist() intervals) since a norm spike can be a single-step
        event that a 100-step sampling interval would miss entirely. Post-clip norm is exactly
        min(grad_norm, max_norm) by construction (that's what clipping does), included for
        convenience rather than because it carries independent information."""
        self.grad_norm_csv.write(
            {
                "step": step,
                "epoch": epoch,
                "grad_norm_pre_clip": grad_norm,
                "grad_norm_post_clip": min(grad_norm, max_norm) if grad_norm == grad_norm else grad_norm,  # NaN-safe
                "clipped": grad_norm > max_norm,
                "max_norm": max_norm,
                "total_loss": total_loss,
            }
        )

    def should_log_dist(self, step: int) -> bool:
        return step % self.dist_interval == 0

    # ---- 1. EMA update verification ----
    @torch.no_grad()
    def maybe_log_ema(self, step: int, epoch: int, teacher, student) -> None:
        if step % self.ema_interval != 0:
            return
        s_params = torch.cat([p.detach().float().reshape(-1) for p in student.parameters()])
        t_params = torch.cat([p.detach().float().reshape(-1) for p in teacher.ema.parameters()])
        student_norm = s_params.norm()
        teacher_norm = t_params.norm()
        l2 = (t_params - s_params).norm()
        rel_l2 = l2 / (student_norm + 1e-12)
        cos = F.cosine_similarity(s_params.unsqueeze(0), t_params.unsqueeze(0)).squeeze(0)
        decay = teacher.decay(teacher.updates) if callable(getattr(teacher, "decay", None)) else None
        self.ema_csv.write(
            {
                "step": step,
                "epoch": epoch,
                "ema_num_updates": teacher.updates,
                "ema_decay": decay,
                "student_parameter_norm": float(student_norm),
                "teacher_parameter_norm": float(teacher_norm),
                "teacher_student_l2": float(l2),
                "teacher_student_relative_l2": float(rel_l2),
                "teacher_student_cosine": float(cos),
            }
        )

    # ---- 2. Training loss decomposition ----
    def log_training_dynamics(
        self,
        step: int,
        epoch: int,
        sup_loss_items,
        unsup_loss_items,
        sup_assign_stats: dict,
        unsup_assign_stats: dict,
        total_loss: float | None = None,
        teacher_conf_max: float | None = None,
        teacher_conf_mean: float | None = None,
        num_teacher_predictions_pre_filter: int | None = None,
        num_teacher_predictions_post_filter: int | None = None,
    ) -> None:
        self.dynamics_csv.write(
            {
                "step": step,
                "epoch": epoch,
                "L_sup_box": float(sup_loss_items[0]),
                "L_sup_cls": float(sup_loss_items[1]),
                "L_sup_dfl": float(sup_loss_items[2]),
                "L_unsup_box": float(unsup_loss_items[0]),
                "L_unsup_cls": float(unsup_loss_items[1]),
                "L_unsup_dfl": float(unsup_loss_items[2]),
                "total_loss": total_loss,
                "num_sup_positive": sup_assign_stats.get("num_sup_positive"),
                "num_unsup_positive": unsup_assign_stats.get("num_unsup_positive"),
                "num_sup_assigned": sup_assign_stats.get("num_sup_assigned"),
                "num_unsup_assigned": unsup_assign_stats.get("num_unsup_assigned"),
                "sum_target_scores_sup": sup_assign_stats.get("sum_target_scores_sup"),
                "sum_target_scores_unsup": unsup_assign_stats.get("sum_target_scores_unsup"),
                "teacher_conf_max": teacher_conf_max,
                "teacher_conf_mean": teacher_conf_mean,
                "num_teacher_predictions_pre_filter": num_teacher_predictions_pre_filter,
                "num_teacher_predictions_post_filter": num_teacher_predictions_post_filter,
            }
        )

    # ---- 4. Assignment statistics ----
    def log_assignment_dynamics(self, step: int, epoch: int, sup_stats: dict, unsup_stats: dict) -> None:
        row = {"step": step, "epoch": epoch}
        for k, v in unsup_stats.items():
            row[f"unsup_{k}"] = v
        for k, v in sup_stats.items():
            row[f"sup_{k}"] = v
        self.assign_csv.write(row)

    # ---- 3, 5, 6. Pseudo-label / DFL / negative-supervision statistics ----
    def maybe_log_pseudo_label_dynamics(
        self,
        step: int,
        epoch: int,
        batch_size: int,
        num_teacher_predictions_pre_filter: int,
        num_pseudo_labels_after_filter: int,
        num_removed_by_confidence: int,
        num_removed_by_nms: int,
        pseudo_conf: torch.Tensor,
        selected_entropy: torch.Tensor,
        rejected_entropy: torch.Tensor,
        negative_supervision_bins: list[dict] | None,
    ) -> None:
        if step % self.dist_interval != 0:
            return
        row = {
            "step": step,
            "epoch": epoch,
            "batch_size": batch_size,
            "num_teacher_predictions_pre_filter": num_teacher_predictions_pre_filter,
            "num_pseudo_labels_after_filter": num_pseudo_labels_after_filter,
            "pseudo_labels_per_image": num_pseudo_labels_after_filter / max(batch_size, 1),
            "num_removed_by_confidence": num_removed_by_confidence,
            "num_removed_by_nms": num_removed_by_nms,
            "pseudo_conf_mean": float(pseudo_conf.mean()) if pseudo_conf.numel() else None,
            "pseudo_conf_std": float(pseudo_conf.std()) if pseudo_conf.numel() > 1 else None,
        }
        row.update({f"pseudo_conf_{k}": v for k, v in _quantiles(pseudo_conf).items()})
        row["selected_dfl_entropy_mean"] = float(selected_entropy.mean()) if selected_entropy.numel() else None
        row["rejected_dfl_entropy_mean"] = float(rejected_entropy.mean()) if rejected_entropy.numel() else None
        row["dfl_entropy_q25"] = float(torch.quantile(selected_entropy.float(), 0.25)) if selected_entropy.numel() else None
        row["dfl_entropy_q50"] = float(torch.quantile(selected_entropy.float(), 0.50)) if selected_entropy.numel() else None
        row["dfl_entropy_q75"] = float(torch.quantile(selected_entropy.float(), 0.75)) if selected_entropy.numel() else None
        if negative_supervision_bins is not None:
            for i, bin_stats in enumerate(negative_supervision_bins):
                lo, hi = i / 10, (i + 1) / 10
                prefix = f"negsup_{lo:.1f}_{hi:.1f}"
                for k, v in bin_stats.items():
                    row[f"{prefix}_{k}"] = v
        self.pseudo_csv.write(row)
