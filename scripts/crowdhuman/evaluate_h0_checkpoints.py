"""Inference-only AP75 and clean-GT edge-error evaluation for H0 checkpoints."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from ultralytics.utils.nms import non_max_suppression
from analyze_dfl_edge_effect import box_iou_one, gt_features, image_id, list_images, load_module, load_odgt, prepare_image, raw_predictions, to_original


def ap(recs: list[tuple[float, int]], n_gt: int) -> float:
    if not recs or not n_gt:
        return float("nan")
    recs.sort(reverse=True)
    tp = np.cumsum([x[1] for x in recs], dtype=float)
    fp = np.cumsum([1 - x[1] for x in recs], dtype=float)
    recall = tp / n_gt
    precision = tp / np.maximum(tp + fp, 1e-12)
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    return float(np.sum((mrec[1:] - mrec[:-1]) * mpre[1:]))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--root", type=Path, default=Path("/work/kawano/LA/datasets/crowdhuman"))
    args = p.parse_args()
    device = torch.device(args.device)
    model = load_module(args.checkpoint, device, "ema")
    images = list_images(args.root, args.root / "val.txt")
    annotations = load_odgt(args.root / "annotation_val.odgt")
    predictions = {0.5: [], 0.75: []}
    edge_px, edge_norm, signed_edge_px, signed_edge_norm, area_ratios = [], [], [], [], []
    n_gt = 0
    n_matched = 0
    for start in range(0, len(images), args.batch_size):
        paths, tensors, meta = images[start:start + args.batch_size], [], []
        for path in paths:
            tensor, ratio, pw, ph, shape = prepare_image(path, 640)
            tensors.append(tensor); meta.append((path, ratio, pw, ph, shape))
        with torch.inference_mode():
            pred, _ = raw_predictions(model, torch.stack(tensors).to(device))
            dets = non_max_suppression(pred, conf_thres=0.001, iou_thres=0.7)
        for det, (path, ratio, pw, ph, (width, height)) in zip(dets, meta):
            gts = [gt_features(x, width, height) for x in annotations.get(image_id(path), [])]
            gt = np.asarray([x["full"] for x in gts], dtype=np.float32)
            n_gt += len(gt)
            if det.numel() == 0:
                continue
            d = det.cpu().numpy()
            boxes = to_original(d[:, :4], ratio, pw, ph, width, height)
            for threshold in predictions:
                used = set()
                for box, score in zip(boxes, d[:, 4]):
                    ious = box_iou_one(box, gt)
                    gi = int(ious.argmax()) if len(ious) else -1
                    hit = gi >= 0 and gi not in used and float(ious[gi]) >= threshold
                    if hit: used.add(gi)
                    predictions[threshold].append((float(score), int(hit)))
            used = set()
            for box in boxes:
                ious = box_iou_one(box, gt)
                gi = int(ious.argmax()) if len(ious) else -1
                if gi < 0 or gi in used or float(ious[gi]) < 0.5:
                    continue
                used.add(gi); n_matched += 1
                denom = np.asarray([gt[gi, 2]-gt[gi, 0], gt[gi, 3]-gt[gi, 1], gt[gi, 2]-gt[gi, 0], gt[gi, 3]-gt[gi, 1]], dtype=np.float32)
                signed = box - gt[gi]  # l/t: positive=inward; r/b: positive=outward.
                errors = np.abs(signed)
                edge_px.extend(errors.tolist()); edge_norm.extend((errors / np.maximum(denom, 1)).tolist())
                signed_edge_px.append(signed)
                signed_edge_norm.append(signed / np.maximum(denom, 1))
                pred_area = max((box[2] - box[0]) * (box[3] - box[1]), 0.0)
                gt_area = max((gt[gi, 2] - gt[gi, 0]) * (gt[gi, 3] - gt[gi, 1]), 1.0)
                area_ratios.append(pred_area / gt_area)
        if start % 400 == 0:
            print(f"{args.checkpoint.name}: {min(start + args.batch_size, len(images))}/{len(images)}", flush=True)
    signed_px = np.asarray(signed_edge_px, dtype=np.float64)
    signed_norm = np.asarray(signed_edge_norm, dtype=np.float64)
    out = {
        "checkpoint": str(args.checkpoint), "images": len(images), "gt_boxes": n_gt,
        "matched_detections_iou50": n_matched, "AP50_custom": ap(predictions[0.5], n_gt),
        "AP75_custom": ap(predictions[0.75], n_gt), "edge_error_px_mean": float(np.mean(edge_px)),
        "edge_error_px_median": float(np.median(edge_px)), "edge_error_norm_mean": float(np.mean(edge_norm)),
        "edge_error_norm_median": float(np.median(edge_norm)), "pred_to_gt_area_ratio_mean": float(np.mean(area_ratios)),
        "pred_to_gt_area_ratio_median": float(np.median(area_ratios)),
    }
    for side, idx in zip(("L", "T", "R", "B"), range(4)):
        out[f"signed_edge_error_px_{side}_mean"] = float(signed_px[:, idx].mean())
        out[f"signed_edge_error_norm_{side}_mean"] = float(signed_norm[:, idx].mean())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out)); w.writeheader(); w.writerow(out)
    print(out, flush=True)


if __name__ == "__main__":
    main()
