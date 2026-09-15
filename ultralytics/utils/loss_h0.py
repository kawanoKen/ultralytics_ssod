"""Fully-supervised artificial-boundary-noise loss for the H0 experiment."""

from __future__ import annotations

import torch

from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.tal import bbox2dist, make_anchors


class H0BoundaryNoiseLoss(v8DetectionLoss):
    """One deterministic noisy GT edge with noisy, masked, or clean-edge DFL targets."""

    def __init__(self, model):
        super().__init__(model)
        h = model.args
        self.noise_fraction = float(h.h0_noise_fraction)
        self.outward_probability = float(getattr(h, "h0_outward_probability", 0.5))
        self.dfl_mode = str(h.h0_dfl_mode)
        self.map_seed = int(h.h0_corruption_seed)
        if self.noise_fraction <= 0 or self.dfl_mode not in {"on", "off", "clean"}:
            raise ValueError("H0 requires positive h0_noise_fraction and h0_dfl_mode in {'on', 'off', 'clean'}")
        if not 0.0 <= self.outward_probability <= 1.0:
            raise ValueError("h0_outward_probability must be in [0, 1]")
        self.last_h0_stats: dict[str, float] = {}

    def _corrupt(self, boxes, labels, valid, imgsz):
        """Return noisy xyxy and its single corrupted side per valid GT.

        Hashing clean, post-augmentation coordinates makes ON/OFF paired runs
        deterministic without process-local RNG.  Geometric augmentation is
        identical within each pair because their model/dataloader seed matches.
        """
        noisy, edge_mask = boxes.clone(), torch.zeros_like(boxes, dtype=torch.bool)
        valid = valid.squeeze(-1)
        if not valid.any():
            self.last_h0_stats = {
                "objects": 0.0, "requested_fraction": 0.0, "applied_fraction": 0.0,
                "clamped_fraction": 0.0, "outward_fraction": 0.0,
            }
            return noisy, edge_mask
        q = torch.round(boxes * 16).to(torch.int64)
        h = (q[..., 0] * 73856093 + q[..., 1] * 19349663 + q[..., 2] * 83492791 + q[..., 3] * 2654435761
             + labels.squeeze(-1).to(torch.int64) * 97531 + self.map_seed * 1000003) & 0x7FFFFFFFFFFFFFFF
        edge = (h % 4).long()
        # Outward is l/t decreasing or r/b increasing; inward is its exact opposite.
        # p=0.5 preserves the legacy symmetric sign hash exactly, allowing the existing High runs
        # to remain the paired 50/50 reference.  Biased conditions draw their direction from a
        # separate deterministic hash component while keeping object, side, and magnitude fixed.
        outward_sign = torch.where(edge < 2, -1.0, 1.0).to(boxes.dtype)
        if self.outward_probability == 0.5:
            sign = torch.where(((h // 4) % 2).bool(), 1.0, -1.0).to(boxes.dtype)
        else:
            direction_u = ((h // 1000003) % 1000003).to(boxes.dtype) / 1000002.0
            sign = torch.where(direction_u < self.outward_probability, outward_sign, -outward_sign)
        # Empirical-scale distribution, fixed before training: Uniform[0.5, 1.5] x scale.
        magnitude = (0.5 + ((h // 8) % 1000003).to(boxes.dtype) / 1000002.0) * self.noise_fraction
        width, height = (boxes[..., 2] - boxes[..., 0]).clamp_min(1), (boxes[..., 3] - boxes[..., 1]).clamp_min(1)
        scale = torch.where((edge == 0) | (edge == 2), width, height)
        delta = sign * magnitude * scale
        before = noisy.clone()
        for side in range(4):
            take = valid & (edge == side)
            if not take.any():
                continue
            if side == 0:
                noisy[..., 0][take] = (noisy[..., 0][take] + delta[take]).clamp(0, imgsz[1] - 1)
                noisy[..., 0][take] = torch.minimum(noisy[..., 0][take], noisy[..., 2][take] - 1)
            elif side == 1:
                noisy[..., 1][take] = (noisy[..., 1][take] + delta[take]).clamp(0, imgsz[0] - 1)
                noisy[..., 1][take] = torch.minimum(noisy[..., 1][take], noisy[..., 3][take] - 1)
            elif side == 2:
                noisy[..., 2][take] = (noisy[..., 2][take] + delta[take]).clamp(1, imgsz[1])
                noisy[..., 2][take] = torch.maximum(noisy[..., 2][take], noisy[..., 0][take] + 1)
            else:
                noisy[..., 3][take] = (noisy[..., 3][take] + delta[take]).clamp(1, imgsz[0])
                noisy[..., 3][take] = torch.maximum(noisy[..., 3][take], noisy[..., 1][take] + 1)
            edge_mask[..., side][take] = True
        applied = (noisy - before).abs().sum(-1)
        self.last_h0_stats = {
            "objects": float(valid.sum()),
            "requested_fraction": float((delta.abs()[valid] / scale[valid]).mean()),
            "applied_fraction": float((applied[valid] / scale[valid]).mean()),
            "clamped_fraction": float((applied[valid] + 1e-3 < delta.abs()[valid]).float().mean()),
            "outward_fraction": float((sign[valid] == outward_sign[valid]).float().mean()),
        }
        return noisy, edge_mask

    def __call__(self, preds, batch):
        loss = torch.zeros(3, device=self.device)
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat([x.view(feats[0].shape[0], self.no, -1) for x in feats], 2).split((self.reg_max * 4, self.nc), 1)
        pred_scores, pred_distri = pred_scores.permute(0, 2, 1).contiguous(), pred_distri.permute(0, 2, 1).contiguous()
        dtype, batch_size = pred_scores.dtype, pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets, batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, clean_boxes = targets.split((1, 4), 2)
        valid = clean_boxes.sum(2, keepdim=True).gt(0.0)
        gt_boxes, corrupt_mask = self._corrupt(clean_boxes, gt_labels, valid, imgsz)
        pred_boxes = self.bbox_decode(anchor_points, pred_distri)
        _, target_boxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(), (pred_boxes.detach() * stride_tensor).type(gt_boxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_boxes, valid,
        )
        target_sum = max(target_scores.sum(), 1)
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_sum
        if fg_mask.sum():
            score_weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
            iou = bbox_iou(pred_boxes[fg_mask], (target_boxes / stride_tensor)[fg_mask], xywh=False, CIoU=True)
            loss[0] = ((1.0 - iou) * score_weight).sum() / target_sum
            # Assignment and IoU intentionally retain the noisy box for every H0 mode.
            # Only ``clean`` replaces the DFL target for the corrupted side with its original GT edge.
            dfl_target_boxes = target_boxes
            if self.dfl_mode == "clean":
                gt_offset = torch.arange(batch_size, device=self.device).unsqueeze(1) * clean_boxes.shape[1]
                flat_gt_idx = target_gt_idx + gt_offset
                dfl_target_boxes = clean_boxes.flatten(0, 1)[flat_gt_idx]
            target_ltrb = bbox2dist(
                anchor_points, dfl_target_boxes / stride_tensor, self.bbox_loss.dfl_loss.reg_max - 1
            )[fg_mask]
            per_edge = self.bbox_loss.dfl_loss(pred_distri[fg_mask].view(-1, self.bbox_loss.dfl_loss.reg_max), target_ltrb)
            if self.dfl_mode == "off":
                # target_gt_idx is local to each image's padded GT table.
                gt_offset = torch.arange(batch_size, device=self.device).unsqueeze(1) * corrupt_mask.shape[1]
                flat_gt_idx = target_gt_idx + gt_offset
                per_edge = per_edge * (~corrupt_mask.flatten(0, 1)[flat_gt_idx[fg_mask]]).to(per_edge.dtype)
            loss[2] = (per_edge.mean(-1, keepdim=True) * score_weight).sum() / target_sum
        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl
        return loss * batch_size, loss.detach()
