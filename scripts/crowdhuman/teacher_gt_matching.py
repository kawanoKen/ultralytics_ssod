"""Teacher-GT matching over training: for each saved teacher checkpoint, match the teacher's
raw NMS-survivor predictions against ground truth on the *unlabeled* split it was actually
pseudo-labeling, and classify every GT box into one of three buckets by the confidence of its
best-matching teacher prediction (IoU >= 0.5):

  q+ reliable : matched prediction has conf >= conf_threshold_high (adopted as a pseudo-label)
  q? ignore   : matched prediction has conf in [conf_threshold_low, conf_threshold_high)
                (excluded from the classification loss via ignore_mask, not trained as positive
                or negative)
  q- missed   : no matching prediction reaches conf_threshold_low at all (silently trained as a
                hard negative -- this is the confirmation-bias failure mode)

Stratified by GT box size (COCO-style area thresholds: small <32^2, medium 32^2-96^2,
large >=96^2, computed in the 640x640 letterboxed pixel space actually fed to the model).

GT is only used here, offline, for this retrospective audit -- never inside training itself.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from ultralytics.data.augment import LetterBox
from ultralytics.utils.nms import non_max_suppression

CONF_THRESHOLD_HIGH = 0.5
CONF_THRESHOLD_LOW = 0.3
IOU_MATCH = 0.5
SMALL_MAX = 32 * 32
MEDIUM_MAX = 96 * 96


def load_gt_boxes_xyxy(label_path: Path, w: int, h: int) -> np.ndarray:
    """Read a YOLO-format label file (cls cx cy w h, normalized) and return absolute xyxy boxes."""
    if not label_path.exists():
        return np.zeros((0, 4), dtype=np.float32)
    rows = np.loadtxt(label_path, dtype=np.float32, ndmin=2)
    if rows.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    cx, cy, bw, bh = rows[:, 1] * w, rows[:, 2] * h, rows[:, 3] * w, rows[:, 4] * h
    x1, y1, x2, y2 = cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2
    return np.stack([x1, y1, x2, y2], axis=1)


def box_iou_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(N,4) x (M,4) xyxy -> (N,M) IoU."""
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


def size_bucket(area: float) -> str:
    if area < SMALL_MAX:
        return "small"
    if area < MEDIUM_MAX:
        return "medium"
    return "large"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, help="e.g. runs/crowdhuman_ssod_1p_ema_fixed/yolov8n_voc_ssod_baseline")
    parser.add_argument("--unlabeled-list", required=True, help="e.g. /work/.../crowdhuman/train_unlabeled_1p.txt")
    parser.add_argument("--epochs", type=int, nargs="+", default=[0, 10, 20, 30, 40, 50, 60, 70, 80])
    parser.add_argument("--n-images", type=int, default=1000)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    list_path = Path(args.unlabeled_list)
    root = list_path.parent
    all_lines = [line.strip() for line in list_path.read_text().splitlines() if line.strip()]
    rng = random.Random(args.seed)
    sample = rng.sample(all_lines, min(args.n_images, len(all_lines)))

    letterbox = LetterBox((args.imgsz, args.imgsz), auto=False, scaleup=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Pre-load images + letterboxed GT once (identical across all epochs).
    import cv2

    samples = []
    for rel in sample:
        img_path = (root / rel).resolve() if not rel.startswith("/") else Path(rel)
        # dataset txt paths look like "./images/train/xxx.jpg" relative to the dataset root
        img_path = (root / rel.lstrip("./")).resolve()
        label_path = Path(str(img_path).replace("/images/", "/labels/")).with_suffix(".txt")
        img0 = cv2.imread(str(img_path))
        if img0 is None:
            continue
        h0, w0 = img0.shape[:2]
        gt_xyxy0 = load_gt_boxes_xyxy(label_path, w0, h0)
        img = letterbox(image=img0)
        r = min(args.imgsz / h0, args.imgsz / w0)
        pad_w = (args.imgsz - w0 * r) / 2
        pad_h = (args.imgsz - h0 * r) / 2
        gt_xyxy = gt_xyxy0 * r
        gt_xyxy[:, [0, 2]] += pad_w
        gt_xyxy[:, [1, 3]] += pad_h
        img_t = torch.from_numpy(img[:, :, ::-1].copy()).permute(2, 0, 1).float() / 255.0
        samples.append((img_t, gt_xyxy))
    print(f"Loaded {len(samples)} images with GT (requested {args.n_images})")

    results_table = {}  # epoch -> size -> {"q+": n, "q?": n, "q-": n, "total": n}

    for epoch in args.epochs:
        ckpt_path = run_dir / "weights" / f"epoch{epoch}.pt"
        if not ckpt_path.exists():
            print(f"skip epoch {epoch}: {ckpt_path} not found")
            continue
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        teacher = ckpt["teacher"].float().to(device).eval()

        counts = {s: {"q+": 0, "q?": 0, "q-": 0, "total": 0} for s in ("small", "medium", "large")}

        with torch.no_grad():
            for img_t, gt_xyxy in samples:
                if gt_xyxy.shape[0] == 0:
                    continue
                pred = teacher(img_t.unsqueeze(0).to(device))
                pred_out = pred[0] if isinstance(pred, tuple) else pred
                dets = non_max_suppression(pred_out, conf_thres=0.01, iou_thres=0.65)[0]  # (n,6) xyxy,conf,cls
                dets_np = dets.cpu().numpy() if dets.numel() else np.zeros((0, 6), dtype=np.float32)

                ious = box_iou_np(gt_xyxy, dets_np[:, :4]) if dets_np.shape[0] else np.zeros((gt_xyxy.shape[0], 0))
                for gi in range(gt_xyxy.shape[0]):
                    area = (gt_xyxy[gi, 2] - gt_xyxy[gi, 0]) * (gt_xyxy[gi, 3] - gt_xyxy[gi, 1])
                    bucket = size_bucket(area)
                    if dets_np.shape[0]:
                        candidate = ious[gi] >= IOU_MATCH
                        best_conf = dets_np[candidate, 4].max() if candidate.any() else 0.0
                    else:
                        best_conf = 0.0
                    if best_conf >= CONF_THRESHOLD_HIGH:
                        counts[bucket]["q+"] += 1
                    elif best_conf >= CONF_THRESHOLD_LOW:
                        counts[bucket]["q?"] += 1
                    else:
                        counts[bucket]["q-"] += 1
                    counts[bucket]["total"] += 1

        results_table[epoch] = counts
        del teacher, ckpt
        torch.cuda.empty_cache()
        line = f"epoch {epoch:>3d}  "
        for s in ("small", "medium", "large"):
            c = counts[s]
            t = max(c["total"], 1)
            line += f"| {s:>6s} q+={c['q+']/t:5.1%} q?={c['q?']/t:5.1%} q-={c['q-']/t:5.1%} (n={c['total']:4d}) "
        print(line)

    out_path = Path(args.out) if args.out else run_dir / "teacher_gt_matching.md"
    with out_path.open("w") as f:
        for size in ("small", "medium", "large"):
            f.write(f"\n## {size}\n\n")
            f.write("| epoch | q+ reliable | q? ignore | q- missed | n |\n")
            f.write("|---|---|---|---|---|\n")
            for epoch in args.epochs:
                if epoch not in results_table:
                    continue
                c = results_table[epoch][size]
                t = max(c["total"], 1)
                f.write(
                    f"| {epoch} | {c['q+'] / t:.1%} | {c['q?'] / t:.1%} | {c['q-'] / t:.1%} | {c['total']} |\n"
                )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
