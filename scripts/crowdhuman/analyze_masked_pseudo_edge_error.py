"""Validation pseudo-edge error for DFL-selected versus non-selected edges.

This is an inference-only analysis.  It applies the same NMS and per-edge DFL
threshold used by the SSOD loss to a saved teacher checkpoint, matches the
surviving reliable pseudo boxes to CrowdHuman full-body GT, and summarizes
pseudo-to-GT edge errors for visible and occluded GT edges.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from ultralytics.utils.nms import non_max_suppression

from analyze_dfl_edge_effect import (
    CONF_HIGH,
    EDGE_THRESHOLD,
    MATCH_IOU,
    NMS_CONF,
    NMS_IOU,
    SIDES,
    box_iou_one,
    edge_conf_from_features,
    gt_features,
    image_id,
    list_images,
    load_module,
    load_odgt,
    prepare_image,
    raw_predictions,
    to_original,
)


def severity(occlusion: float) -> str:
    return "visible" if not math.isfinite(occlusion) or occlusion <= 0.05 else "occluded"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/work/kawano/LA/datasets/crowdhuman"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    images = list_images(args.root, args.root / "val.txt")
    annotations = load_odgt(args.root / "annotation_val.odgt")
    model = load_module(args.checkpoint, device, "teacher")
    reg_max = int(model.model[-1].reg_max)
    rows: list[dict[str, object]] = []
    unmatched = 0

    for start in range(0, len(images), args.batch_size):
        paths = images[start : start + args.batch_size]
        tensors, meta = [], []
        for path in paths:
            tensor, r, pw, ph, shape = prepare_image(path, args.imgsz)
            tensors.append(tensor)
            meta.append((path, r, pw, ph, shape))
        batch = torch.stack(tensors).to(device)
        with torch.inference_mode():
            pred, feats = raw_predictions(model, batch)
            labels, keeps = non_max_suppression(pred, conf_thres=NMS_CONF, iou_thres=NMS_IOU, return_idxs=True)

        for bi, (path, r, pw, ph, (w, h)) in enumerate(meta):
            dets = labels[bi]
            if dets.numel() == 0:
                continue
            det_np = dets.cpu().numpy()
            boxes = to_original(det_np[:, :4], r, pw, ph, w, h)
            edge_conf, _ = edge_conf_from_features(feats, keeps[bi], bi, reg_max, model.model[-1].stride)
            gts = [gt_features(x, w, h) for x in annotations.get(image_id(path), [])]
            gt_boxes = np.asarray([x["full"] for x in gts], dtype=np.float32) if gts else np.zeros((0, 4), dtype=np.float32)
            for di, det in enumerate(det_np):
                if float(det[4]) < CONF_HIGH or gt_boxes.shape[0] == 0:
                    continue
                ious = box_iou_one(boxes[di], gt_boxes)
                gt_idx = int(np.argmax(ious))
                if float(ious[gt_idx]) < MATCH_IOU:
                    unmatched += 1
                    continue
                gt = gts[gt_idx]
                for side_idx, side in enumerate(SIDES):
                    occ = float(gt["occ"][side_idx])
                    pred_edge = float(boxes[di, side_idx])
                    gt_edge = float(gt["full"][side_idx])
                    rows.append(
                        {
                            "image_id": image_id(path),
                            "side": side,
                            "mask_status": "DFL-masked" if edge_conf[di, side_idx] < EDGE_THRESHOLD else "non-masked",
                            "visibility": severity(occ),
                            "occlusion": occ,
                            "edge_confidence": float(edge_conf[di, side_idx]),
                            "classification_confidence": float(det[4]),
                            "pseudo_gt_iou": float(ious[gt_idx]),
                            "edge_error_px": abs(pred_edge - gt_edge),
                        }
                    )
        if (start // args.batch_size) % 20 == 0:
            print(f"processed {min(start + args.batch_size, len(images))}/{len(images)} validation images", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        fieldnames = ["visibility", "mask_status", "side", "n", "mean_error_px", "median_error_px", "mean_edge_confidence", "mean_occlusion"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        groups = defaultdict(list)
        for row in rows:
            groups[(row["visibility"], row["mask_status"], row["side"])].append(row)
        for visibility in ("visible", "occluded"):
            for mask_status in ("DFL-masked", "non-masked"):
                for side in SIDES:
                    group = groups[(visibility, mask_status, side)]
                    errors = [float(x["edge_error_px"]) for x in group]
                    writer.writerow(
                        {
                            "visibility": visibility,
                            "mask_status": mask_status,
                            "side": side,
                            "n": len(group),
                            "mean_error_px": float(np.mean(errors)) if errors else np.nan,
                            "median_error_px": float(np.median(errors)) if errors else np.nan,
                            "mean_edge_confidence": float(np.mean([x["edge_confidence"] for x in group])) if group else np.nan,
                            "mean_occlusion": float(np.mean([x["occlusion"] for x in group])) if group else np.nan,
                        }
                    )
    print(f"wrote {args.out} ({len(rows)} matched edge rows; unmatched reliable boxes={unmatched})", flush=True)


if __name__ == "__main__":
    main()
