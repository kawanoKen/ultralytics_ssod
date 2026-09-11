"""Isolates whether the huge unsupervised gradient seen at low pseudo-label counts (1%/10%
CrowdHuman) is caused by the pseudo-label content/model state itself, or by
EfficientTeacherLoss's classification-loss normalization (dividing by
max(sum_target_scores_unsup, 1)).

For each fixed batch, records the full pseudo-label/assignment/loss/gradient picture (section 1
of the request), AND three classification-loss normalization counterfactuals built from the
exact same BCE numerator and model state (so only the divisor differs):

  A. current   : divide by target_scores_sum_reliable (= max(sum_target_scores_unsup, 1))
  B. fixed     : divide by a content-independent reference (n_images * anchors_per_image)
  C. count-based: divide by max(num_reliable_pseudo, 1)

Box/DFL loss terms are left untouched in all three -- only the classification term's divisor
changes, achieved by substituting into the SAME differentiable loss vector EfficientTeacherLoss
already returned (so box/dfl stay bit-identical, sharing the same upstream graph).

No optimizer update; gradients are the raw (pre-clipping) autograd.grad output.
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import cv2
import numpy as np
import torch

from ultralytics.data.augment import LetterBox
from ultralytics.models.yolo.detect.ssod_train import xyxy_to_xywh
from ultralytics.utils.loss_ssod import EfficientTeacherLoss
from ultralytics.utils.nms import non_max_suppression

CONF_THRESHOLD_HIGH = 0.5
CONF_THRESHOLD_LOW = 0.3
IOU_MATCH = 0.5
SSOD_WEIGHT = 0.5
ANCHORS_PER_IMAGE = 80 * 80 + 40 * 40 + 20 * 20  # 640 input, strides 8/16/32


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
    if pred_xyxy.shape[0] == 0 or gt_xyxy.shape[0] == 0:
        return 0
    ious = box_iou_np(pred_xyxy, gt_xyxy)
    used_gt = np.zeros(gt_xyxy.shape[0], dtype=bool)
    tp = 0
    for i in np.argsort(-ious.max(axis=1)):
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


def grad_vec(loss_scalar, params, retain_graph):
    g = torch.autograd.grad(loss_scalar, params, retain_graph=retain_graph, allow_unused=True)
    return torch.cat([gi.reshape(-1) for gi, p in zip(g, params) if gi is not None])


def cos(a, b):
    return (torch.dot(a, b) / max(a.norm().item() * b.norm().item(), 1e-12)).item()


def analyze_one_batch(student, teacher, labeled_imgs, labeled_gts, unlabeled_imgs, unlabeled_gts, device):
    n_images = unlabeled_imgs.shape[0]
    labeled_batch = {"img": labeled_imgs.to(device), **make_labeled_target_dict(labeled_gts, device)}

    preds_labeled = student(labeled_batch["img"])
    loss_sup_vec, loss_items_sup = student.loss(labeled_batch, preds_labeled)
    loss_sup_scalar = loss_sup_vec.sum()

    with torch.no_grad():
        pred_out = teacher(unlabeled_imgs.to(device))
    preds_teacher = pred_out[0] if isinstance(pred_out, tuple) else pred_out
    nms_out = non_max_suppression(preds_teacher.detach(), conf_thres=0.01, iou_thres=0.65)

    batch_idx_list, bbox_list, conf_list, cls_list = [], [], [], []
    tp_total, pred_total, gt_total, conf_sum = 0, 0, 0, 0.0
    for img_idx, dets in enumerate(nms_out):
        dets_np = dets.cpu().numpy() if dets.numel() else np.zeros((0, 6), dtype=np.float32)
        reliable = dets_np[dets_np[:, 4] >= CONF_THRESHOLD_HIGH] if dets_np.shape[0] else dets_np
        gt = unlabeled_gts[img_idx]
        tp_total += greedy_match_count(reliable[:, :4], gt, IOU_MATCH)
        pred_total += reliable.shape[0]
        gt_total += gt.shape[0]
        conf_sum += reliable[:, 4].sum() if reliable.shape[0] else 0.0
        if dets.numel():
            n = dets.shape[0]
            batch_idx_list.append(torch.full((n, 1), img_idx, device=device, dtype=torch.long))
            bbox_list.append(dets[:, :4])
            conf_list.append(dets[:, 4:5])
            cls_list.append(dets[:, 5:6])

    pseudo_precision = tp_total / max(pred_total, 1)
    pseudo_recall = tp_total / max(gt_total, 1)
    pseudo_labels_per_image = pred_total / n_images
    mean_confidence = conf_sum / max(pred_total, 1)

    unlabeled_batch_idx = torch.cat(batch_idx_list)
    unlabeled_bboxes = xyxy_to_xywh(torch.cat(bbox_list)) / unlabeled_imgs.shape[2]
    unlabeled_conf = torch.cat(conf_list)
    unlabeled_cls = torch.cat(cls_list)

    preds_unlabeled = student(unlabeled_imgs.to(device))
    loss_func = EfficientTeacherLoss(
        student,
        conf_threshold_high=CONF_THRESHOLD_HIGH,
        conf_threshold_low=CONF_THRESHOLD_LOW,
        use_loc_conf=False,
        use_edge_conf=False,
        assignment_stability_method="off",
    )
    loss_unsup_vec, loss_items_unsup, _, _ = loss_func(
        preds_unlabeled,
        unlabeled_bboxes,
        unlabeled_cls,
        unlabeled_conf,
        unlabeled_batch_idx,
        compute_extra_diag=True,
    )
    assign_stats = loss_func.last_assignment_stats
    numerator = loss_func.last_cls_loss_numerator  # differentiable, un-normalized BCE sum
    s_t = loss_func.last_target_scores_sum_reliable  # max(sum_target_scores_unsup, 1), as actually used
    cls_gain = loss_func.hyp.cls
    n_reliable = assign_stats.get("num_pseudo_boxes", 0)

    # loss_unsup_vec = (box, cls, dfl) already * batch_size (see EfficientTeacherLoss.__call__'s
    # `return loss * batch_size, ...`); box/dfl terms are reused as-is for every variant.
    box_term, cls_term_A, dfl_term = loss_unsup_vec[0], loss_unsup_vec[1], loss_unsup_vec[2]
    batch_size = preds_unlabeled[0].shape[0] if isinstance(preds_unlabeled, (list, tuple)) else preds_unlabeled.shape[0]

    fixed_denom = max(n_images * ANCHORS_PER_IMAGE, 1)
    count_denom = max(n_reliable, 1)
    cls_term_B = numerator / fixed_denom * cls_gain * batch_size
    cls_term_C = numerator / count_denom * cls_gain * batch_size

    loss_A = box_term + cls_term_A + dfl_term
    loss_B = box_term + cls_term_B + dfl_term
    loss_C = box_term + cls_term_C + dfl_term

    params = [p for p in student.parameters() if p.requires_grad]
    flat_s = grad_vec(loss_sup_scalar, params, retain_graph=True)
    flat_uA = grad_vec(loss_A, params, retain_graph=True)
    flat_uB = grad_vec(loss_B, params, retain_graph=True)
    flat_uC = grad_vec(loss_C, params, retain_graph=False)

    norm_s = flat_s.norm().item()
    result = {
        "pseudo_precision": pseudo_precision,
        "pseudo_recall": pseudo_recall,
        "num_reliable_pseudo": n_reliable,
        "pseudo_labels_per_image": pseudo_labels_per_image,
        "num_positive_anchors": assign_stats.get("num_unsup_positive", 0),
        "sum_target_scores_unsup": assign_stats.get("sum_target_scores_unsup", 0.0),
        "target_scores_sum_used": s_t,
        "mean_reliable_confidence": mean_confidence,
        "L_sup_box": float(loss_items_sup[0]),
        "L_sup_cls": float(loss_items_sup[1]),
        "L_sup_dfl": float(loss_items_sup[2]),
        "L_unsup_box": float(loss_items_unsup[0]),
        "L_unsup_cls": float(loss_items_unsup[1]),
        "L_unsup_dfl": float(loss_items_unsup[2]),
        "norm_s": norm_s,
    }
    for tag, flat_u in (("A_current", flat_uA), ("B_fixed", flat_uB), ("C_count", flat_uC)):
        norm_u = flat_u.norm().item()
        flat_total = flat_s + SSOD_WEIGHT * flat_u
        result[f"norm_u_{tag}"] = norm_u
        result[f"ratio_u_over_s_{tag}"] = norm_u / max(norm_s, 1e-12)
        result[f"lambda_ratio_{tag}"] = SSOD_WEIGHT * norm_u / max(norm_s, 1e-12)
        result[f"cos_su_{tag}"] = cos(flat_s, flat_u)
        result[f"cos_total_{tag}"] = cos(flat_total, flat_s)

    student.zero_grad(set_to_none=True)
    return result


def analyze_epoch(run_dir: Path, epoch: int, labeled_list: Path, unlabeled_list: Path, n_images: int, n_batches: int, device):
    ckpt = torch.load(run_dir / "weights" / f"epoch{epoch}.pt", map_location="cpu", weights_only=False)
    student = ckpt["ema"].float().to(device)
    student.criterion = None
    for p in student.parameters():
        p.requires_grad_(True)
    student.train()
    teacher = ckpt["teacher"].float().to(device).eval()

    rows = []
    for b in range(n_batches):
        labeled_imgs, labeled_gts = load_batch(labeled_list, n_images, 640, seed=b)
        unlabeled_imgs, unlabeled_gts = load_batch(unlabeled_list, n_images, 640, seed=1000 + b)
        row = analyze_one_batch(student, teacher, labeled_imgs, labeled_gts, unlabeled_imgs, unlabeled_gts, device)
        row["epoch"] = epoch
        row["batch"] = b
        rows.append(row)
        torch.cuda.empty_cache()
    del student, teacher, ckpt
    torch.cuda.empty_cache()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--labeled-list", required=True)
    parser.add_argument("--unlabeled-list", required=True)
    parser.add_argument("--epochs", type=int, nargs="+", default=[10, 30, 50, 70, 90])
    parser.add_argument("--n-images", type=int, default=32)
    parser.add_argument("--n-batches", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    all_rows = []
    for epoch in args.epochs:
        rows = analyze_epoch(
            Path(args.run_dir), epoch, Path(args.labeled_list), Path(args.unlabeled_list), args.n_images, args.n_batches, device
        )
        all_rows += rows
        means = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k not in ("epoch", "batch")}
        print(epoch, {k: round(v, 4) for k, v in means.items() if "cos" in k or "ratio" in k or k in ("sum_target_scores_unsup", "num_reliable_pseudo")})

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Wrote {args.out} ({len(all_rows)} rows)")


if __name__ == "__main__":
    main()
