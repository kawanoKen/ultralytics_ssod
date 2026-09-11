"""Original BN vs BN-recalibrated validation performance, for a saved SSOD checkpoint.

Extends bn_mismatch_diagnosis2.py's confidence/NMS-survivor comparison (which stopped at raw
detection statistics) all the way through to actual validation metrics (P/R/mAP50/mAP50-95),
using the real ultralytics DetectionValidator so the numbers are directly comparable to what's
already in results.csv. No training is performed -- BN running stats are recomputed via forward
passes only (train()-mode BN with reset_running_stats(), no backward/optimizer step).
"""

from __future__ import annotations

import argparse
import copy
import random
from pathlib import Path

import cv2
import torch
from torch import nn

from ultralytics import YOLO
from ultralytics.data.augment import LetterBox


def load_calib_batch(list_path: Path, n: int, imgsz: int, seed: int) -> torch.Tensor:
    root = list_path.parent
    lines = [x.strip() for x in list_path.read_text().splitlines() if x.strip()]
    rng = random.Random(seed)
    sample = rng.sample(lines, min(n, len(lines)))
    lb = LetterBox((imgsz, imgsz), auto=False, scaleup=True)
    imgs = []
    for rel in sample:
        p = (root / rel.lstrip("./")).resolve()
        img0 = cv2.imread(str(p))
        if img0 is None:
            continue
        img = lb(image=img0)
        imgs.append(torch.from_numpy(img[:, :, ::-1].copy()).permute(2, 0, 1).float() / 255.0)
    return torch.stack(imgs, 0)


def recalibrate_bn(model: nn.Module, calib_batches: list[torch.Tensor], device: torch.device) -> None:
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats()
            m.momentum = None  # cumulative average over all calibration batches
    model.to(device).train()
    with torch.no_grad():
        for b in calib_batches:
            model(b.to(device))
    model.eval()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="e.g. runs/.../weights/best.pt")
    parser.add_argument("--unlabeled-list", required=True, help="recalibration images (real, unlabeled split)")
    parser.add_argument("--data", required=True, help="dataset yaml for .val()")
    parser.add_argument("--n-calib-images", type=int, default=32)
    parser.add_argument("--n-calib-batches", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--out", default=None, help="path to save the recalibrated checkpoint")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(args.checkpoint)
    out_path = Path(args.out) if args.out else ckpt_path.with_name(ckpt_path.stem + "_bn_recalibrated.pt")

    print("=== Original BN: validating as-is ===")
    metrics_orig = YOLO(str(ckpt_path)).val(data=args.data, device=args.device, imgsz=args.imgsz, batch=args.batch, split="val")

    print("\n=== Recalibrating BN on real unlabeled images ===")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = copy.deepcopy(ckpt["model"]).float()
    calib_batches = [
        load_calib_batch(Path(args.unlabeled_list), args.n_calib_images, args.imgsz, seed=100 + i)
        for i in range(args.n_calib_batches)
    ]
    recalibrate_bn(model, calib_batches, device)
    ckpt["model"] = model.cpu().half()
    torch.save(ckpt, out_path)
    print(f"Saved recalibrated checkpoint to {out_path}")

    print("\n=== Recalibrated BN: validating ===")
    metrics_recal = YOLO(str(out_path)).val(data=args.data, device=args.device, imgsz=args.imgsz, batch=args.batch, split="val")

    def summarize(m):
        return {
            "precision": float(m.box.mp),
            "recall": float(m.box.mr),
            "mAP50": float(m.box.map50),
            "mAP50-95": float(m.box.map),
        }

    print("\n=== Summary ===")
    orig, recal = summarize(metrics_orig), summarize(metrics_recal)
    for k in orig:
        print(f"{k:10s}: original={orig[k]:.4f}  recalibrated={recal[k]:.4f}  delta={recal[k]-orig[k]:+.4f}")


if __name__ == "__main__":
    main()
