"""Offline (no new training) analysis reconstructing, from saved checkpoints + real data:

  1. Pseudo-label precision/recall against ground truth (GT is only used here, retrospectively,
     never during training) and pseudo-labels/image, for the teacher's *adopted* (reliable,
     conf >= conf_threshold_high) boxes.
  2. Supervised-vs-unsupervised gradient geometry: ||g_u||/||g_s|| and cos(g_s, g_u), via
     ultralytics.utils.loss_spike_lab.compute_split_gradients on one reconstructed labeled +
     unlabeled forward/loss pass.

Used to compare the 1% and 10% CrowdHuman SSOD skip runs on equal footing.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import torch

from ultralytics.data.augment import LetterBox
from ultralytics.models.yolo.detect.ssod_train import xyxy_to_xywh
from ultralytics.utils.loss_spike_lab import compute_split_gradients
from ultralytics.utils.loss_ssod import EfficientTeacherLoss
from ultralytics.utils.nms import non_max_suppression

CONF_THRESHOLD_HIGH = 0.5
CONF_THRESHOLD_LOW = 0.3
IOU_MATCH = 0.5


def load_gt_xyxy(label_path: Path, w: int, h: int) -> np.ndarray:
    if not label_path.exists():
        return np.zeros((0, 4), dtype=np.float32)
    rows = np.loadtxt(label_path, dtype=np.float32, ndmin=2)
    if rows.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    cx, cy, bw, bh = rows[:, 1] * w, rows[:, 2] * h, rows[:, 3] * w, rows[:, 4] * h
    x1, y1, x2, y2 = cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2
    return np.stack([x1, y1, x2, y2], axis=1)


def box_iou_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.clip(union, 1e-9, None)


def greedy_match_count(pred_xyxy: np.ndarray, gt_xyxy: np.ndarray, iou_thr: float) -> int:
    """Standard one-to-one greedy TP count (each GT matched at most once)."""
    if pred_xyxy.shape[0] == 0 or gt_xyxy.shape[0] == 0:
        return 0
    ious = box_iou_np(pred_xyxy, gt_xyxy)
    used_gt = np.zeros(gt_xyxy.shape[0], dtype=bool)
    tp = 0
    order = np.argsort(-ious.max(axis=1))  # process predictions best-first
    for i in order:
        j = np.argmax(ious[i])
        if ious[i, j] >= iou_thr and not used_gt[j]:
            used_gt[j] = True
            tp += 1
    return tp


def load_batch(list_path: Path, n: int, imgsz: int, seed: int):
    root = list_path.parent
    lines = [x.strip() for x in list_path.read_text().splitlines() if x.strip()]
    rng = random.Random(seed)
    sample = rng.sample(lines, min(n, len(lines)))
    lb = LetterBox((imgsz, imgsz), auto=False, scaleup=True)
    imgs, gts = [], []
    for rel in sample:
        img_path = (root / rel.lstrip("./")).resolve()
        label_path = Path(str(img_path).replace("/images/", "/labels/")).with_suffix(".txt")
        img0 = cv2.imread(str(img_path))
        if img0 is None:
            continue
        h0, w0 = img0.shape[:2]
        gt0 = load_gt_xyxy(label_path, w0, h0)
        img = lb(image=img0)
        r = min(imgsz / h0, imgsz / w0)
        pad_w, pad_h = (imgsz - w0 * r) / 2, (imgsz - h0 * r) / 2
        gt = gt0 * r
        gt[:, [0, 2]] += pad_w
        gt[:, [1, 3]] += pad_h
        imgs.append(torch.from_numpy(img[:, :, ::-1].copy()).permute(2, 0, 1).float() / 255.0)
        gts.append(gt)
    return torch.stack(imgs, 0), gts


def make_labeled_target_dict(gts: list[np.ndarray], device) -> dict:
    """Build the {'batch_idx','cls','bboxes'} dict v8DetectionLoss.preprocess expects, from pixel
    xyxy GT boxes in a 640x640 letterboxed image (single class: person, cls=0)."""
    batch_idx, cls, bboxes = [], [], []
    for i, gt in enumerate(gts):
        if gt.shape[0] == 0:
            continue
        xywh = xyxy_to_xywh(torch.from_numpy(gt).float()) / 640.0
        bboxes.append(xywh)
        cls.append(torch.zeros(gt.shape[0], 1))
        batch_idx.append(torch.full((gt.shape[0],), i, dtype=torch.float32))
    return {
        "batch_idx": torch.cat(batch_idx).to(device) if batch_idx else torch.zeros(0, device=device),
        "cls": torch.cat(cls).to(device) if cls else torch.zeros(0, 1, device=device),
        "bboxes": torch.cat(bboxes).to(device) if bboxes else torch.zeros(0, 4, device=device),
    }


def analyze(run_dir: Path, epoch: int, labeled_list: Path, unlabeled_list: Path, n_images: int, device, seed: int):
    ckpt = torch.load(run_dir / "weights" / f"epoch{epoch}.pt", map_location="cpu", weights_only=False)
    student = ckpt["ema"].float().to(device)
    # The checkpoint carries a stale `criterion` (v8DetectionLoss) cached from training -- it's a
    # plain Python attribute, not a registered nn.Module buffer, so `.to(device)` never moves its
    # internal tensors (e.g. the DFL `proj` vector), leaving it on whatever device torch.load put
    # it on. Drop it so `.loss()` recreates it fresh on the correct device.
    student.criterion = None
    # ModelEMA explicitly sets requires_grad=False on every EMA parameter (it's normally an
    # inference-only copy) -- re-enable it here so the reconstructed forward pass actually
    # produces a differentiable graph for compute_split_gradients to read.
    for p in student.parameters():
        p.requires_grad_(True)
    student.train()  # matches self._model_train() during the real forward this reconstructs
    teacher = ckpt["teacher"].float().to(device).eval()
    nc = 1

    labeled_imgs, labeled_gts = load_batch(labeled_list, n_images, 640, seed)
    unlabeled_imgs, unlabeled_gts = load_batch(unlabeled_list, n_images, 640, seed + 1)
    labeled_batch = {"img": labeled_imgs.to(device), **make_labeled_target_dict(labeled_gts, device)}

    # Supervised loss
    preds_labeled = student(labeled_batch["img"])
    loss_sup, _ = student.loss(labeled_batch, preds_labeled)
    loss_sup = loss_sup.sum()

    # Teacher pseudo-labels on unlabeled images
    with torch.no_grad():
        pred_out = teacher(unlabeled_imgs.to(device))
    preds_teacher = pred_out[0] if isinstance(pred_out, tuple) else pred_out
    nms_out = non_max_suppression(preds_teacher.detach(), conf_thres=0.01, iou_thres=0.65)

    batch_idx_list, bbox_list, conf_list, cls_list = [], [], [], []
    tp_total, pred_total, gt_total = 0, 0, 0
    for img_idx, dets in enumerate(nms_out):
        dets_np = dets.cpu().numpy() if dets.numel() else np.zeros((0, 6), dtype=np.float32)
        reliable = dets_np[dets_np[:, 4] >= CONF_THRESHOLD_HIGH] if dets_np.shape[0] else dets_np
        gt = unlabeled_gts[img_idx]
        tp_total += greedy_match_count(reliable[:, :4], gt, IOU_MATCH)
        pred_total += reliable.shape[0]
        gt_total += gt.shape[0]
        if dets.numel():
            n = dets.shape[0]
            batch_idx_list.append(torch.full((n, 1), img_idx, device=device, dtype=torch.long))
            bbox_list.append(dets[:, :4])
            conf_list.append(dets[:, 4:5])
            cls_list.append(dets[:, 5:6])

    pseudo_precision = tp_total / max(pred_total, 1)
    pseudo_recall = tp_total / max(gt_total, 1)
    pseudo_labels_per_image = pred_total / len(nms_out)

    unlabeled_batch_idx = torch.cat(batch_idx_list)
    unlabeled_bboxes = xyxy_to_xywh(torch.cat(bbox_list)) / unlabeled_imgs.shape[2]
    unlabeled_conf = torch.cat(conf_list)
    unlabeled_cls = torch.cat(cls_list)

    student.zero_grad()
    preds_unlabeled = student(unlabeled_imgs.to(device))
    loss_func = EfficientTeacherLoss(
        student,
        conf_threshold_high=CONF_THRESHOLD_HIGH,
        conf_threshold_low=CONF_THRESHOLD_LOW,
        use_loc_conf=False,
        use_edge_conf=False,
        assignment_stability_method="off",
    )
    loss_unsup, _, _, _ = loss_func(preds_unlabeled, unlabeled_bboxes, unlabeled_cls, unlabeled_conf, unlabeled_batch_idx)
    loss_unsup = loss_unsup.sum()

    norm_s, norm_u, cos_su, r_t = compute_split_gradients(student, loss_sup, 0.5 * loss_unsup)

    return {
        "pseudo_precision": pseudo_precision,
        "pseudo_recall": pseudo_recall,
        "pseudo_labels_per_image": pseudo_labels_per_image,
        "grad_norm_ratio_u_over_s": r_t,
        "cos_su": cos_su,
        "norm_s": norm_s,
        "norm_u": norm_u,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--labeled-list", required=True)
    parser.add_argument("--unlabeled-list", required=True)
    parser.add_argument("--epoch", type=int, default=90)
    parser.add_argument("--n-images", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    result = analyze(
        Path(args.run_dir), args.epoch, Path(args.labeled_list), Path(args.unlabeled_list), args.n_images, device, args.seed
    )
    for k, v in result.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
