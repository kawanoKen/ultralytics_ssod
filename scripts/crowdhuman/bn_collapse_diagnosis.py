"""Four-way comparison to localize the source of the near-zero teacher confidence observed
when loading saved SSOD checkpoints in eval() mode (see teacher_gt_matching.py investigation).
No additional training is performed -- this only re-runs inference on already-saved checkpoints.

Conditions, same checkpoint epoch and same image batch throughout:
  1. teacher_eval              : ckpt['teacher'] in eval() -- what pseudo-labeling actually uses.
  2. proxy_student_eval        : ckpt['ema'] in eval() -- BaseTrainer's own EMA of the student.
                                  NOTE: the raw (non-EMA) student is never checkpointed
                                  ("model": None by design, see engine/trainer.py save_model),
                                  so this is the closest available proxy, not the literal raw
                                  student. With burn_in_epochs=0 both EMAs started from the same
                                  point, so if this ALSO collapses it rules out anything
                                  teacher-instance-specific.
  3. teacher_with_student_bn   : ckpt['teacher']'s conv/head weights, but with every BatchNorm2d's
                                  running_mean/running_var/num_batches_tracked overwritten by the
                                  corresponding buffers from ckpt['ema']. Isolates whether it is
                                  specifically the teacher's OWN BN buffers that are bad.
  4. teacher_bn_recalibrated   : ckpt['teacher'] with all BatchNorm2d running stats reset and
                                  recomputed from scratch by forward-passing real (weak-aug-free,
                                  letterboxed) images in train() mode, then switched back to
                                  eval() for the actual measurement forward pass. Tests whether
                                  simply re-estimating the running statistics against current
                                  data restores normal output, independent of anything else.

For each condition we report: max confidence, confidence quantiles, NMS survivor count, and
raw logit quantiles (logit = inverse-sigmoid of the reported confidence channel).
"""

from __future__ import annotations

import argparse
import copy
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from ultralytics.data.augment import LetterBox
from ultralytics.utils.nms import non_max_suppression

QUANTILES = (0.5, 0.9, 0.99, 0.999, 1.0)


def load_batch(list_path: Path, n: int, imgsz: int, seed: int = 0) -> torch.Tensor:
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


def report(name: str, model: nn.Module, batch: torch.Tensor, device: torch.device) -> None:
    model = model.to(device).eval()
    with torch.no_grad():
        pred = model(batch.to(device))
    p0 = pred[0] if isinstance(pred, tuple) else pred
    conf = p0[:, 4, :].flatten().clamp(1e-12, 1 - 1e-12)
    logit = torch.logit(conf)
    dets_per_img = [
        d.shape[0] for d in non_max_suppression(p0, conf_thres=0.01, iou_thres=0.65)
    ]
    conf_q = torch.quantile(conf.float(), torch.tensor(QUANTILES))
    logit_q = torch.quantile(logit.float(), torch.tensor(QUANTILES))
    print(f"\n=== {name} ===")
    print(f"  max_conf={conf.max().item():.6g}")
    print("  conf quantiles  " + " ".join(f"p{int(q*100)}={v:.4g}" for q, v in zip(QUANTILES, conf_q)))
    print("  logit quantiles " + " ".join(f"p{int(q*100)}={v:.4g}" for q, v in zip(QUANTILES, logit_q)))
    print(f"  NMS survivors per image: mean={np.mean(dets_per_img):.2f} max={max(dets_per_img)} nonzero_imgs={sum(d>0 for d in dets_per_img)}/{len(dets_per_img)}")


def copy_bn_buffers(dst: nn.Module, src: nn.Module) -> None:
    dst_bns = [m for m in dst.modules() if isinstance(m, nn.BatchNorm2d)]
    src_bns = [m for m in src.modules() if isinstance(m, nn.BatchNorm2d)]
    assert len(dst_bns) == len(src_bns), f"BN count mismatch: {len(dst_bns)} vs {len(src_bns)}"
    with torch.no_grad():
        for d, s in zip(dst_bns, src_bns):
            d.running_mean.copy_(s.running_mean)
            d.running_var.copy_(s.running_var)
            d.num_batches_tracked.copy_(s.num_batches_tracked)


def recalibrate_bn(model: nn.Module, calib_batches: list[torch.Tensor], device: torch.device) -> None:
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats()
            m.momentum = None  # cumulative moving average over all calibration batches
    model.to(device).train()
    with torch.no_grad():
        for b in calib_batches:
            model(b.to(device))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--unlabeled-list", required=True)
    parser.add_argument("--n-images", type=int, default=32)
    parser.add_argument("--n-calib-batches", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cuda:2")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    print(f"epoch={ckpt['epoch']} teacher_updates={ckpt['teacher_updates']} ema_updates={ckpt['updates']}")

    list_path = Path(args.unlabeled_list)
    test_batch = load_batch(list_path, args.n_images, args.imgsz, seed=0)
    print(f"test batch: {test_batch.shape}")

    teacher = ckpt["teacher"].float()
    proxy_student = ckpt["ema"].float()

    # 1. teacher eval
    report("1. teacher_eval (actual pseudo-labeling path)", copy.deepcopy(teacher), test_batch, device)

    # 2. proxy student (BaseTrainer's own EMA) eval
    report("2. proxy_student_eval (ckpt['ema']; raw student is never checkpointed)", copy.deepcopy(proxy_student), test_batch, device)

    # 3. teacher weights + student's BN buffers
    hybrid = copy.deepcopy(teacher)
    copy_bn_buffers(hybrid, proxy_student)
    report("3. teacher_with_student_bn (teacher conv/head weights + ema's BN running stats)", hybrid, test_batch, device)

    # 4. teacher with BN recalibrated on real data
    recal = copy.deepcopy(teacher)
    calib_batches = [load_batch(list_path, args.n_images, args.imgsz, seed=100 + i) for i in range(args.n_calib_batches)]
    recalibrate_bn(recal, calib_batches, device)
    report(f"4. teacher_bn_recalibrated ({args.n_calib_batches} calibration batches of {args.n_images} real images)", recal, test_batch, device)


if __name__ == "__main__":
    main()
