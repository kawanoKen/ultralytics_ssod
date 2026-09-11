"""Second-pass BN mismatch diagnosis for the crowdhuman_ssod_1p_ema_fixed collapse, per the
A/B/C/D protocol:

  A. EMA Teacher weight + Teacher's own (stored) BN buffers          -- the actual pseudo-label path
  B. EMA Teacher weight + current Student(proxy) BN buffers copied in
  C. EMA Teacher weight + BN recalibrated on real current images
  D. Student(proxy) weight + Student(proxy)'s own BN buffers

NOTE: the raw (non-EMA) student is never checkpointed ("model": None by design in
engine/trainer.py save_model -- "resume and final checkpoints derive from EMA"), so
ckpt['ema'] (BaseTrainer's own separately-tracked EMA of the student) is used as the closest
available stand-in for "Student" in B and D, as already flagged in the prior report.

For each condition: max/mean/median confidence, p90/p99 confidence, NMS survivors/image,
raw-logit median/p1/p99.

Additionally, for every BatchNorm2d layer in both the Teacher and the Student-proxy model,
compares the STORED running_mean/running_var against mean/var freshly estimated from the same
32-image batch's actual activations at that layer (via forward hooks), to localize which layers
diverge most.

No training is performed; only existing checkpoints are read.
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

CONF_QUANTILES = (0.5, 0.9, 0.99)
LOGIT_QUANTILES = (0.01, 0.5, 0.99)


def load_batch(list_path: Path, n: int, imgsz: int, seed: int) -> torch.Tensor:
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
    dets_per_img = [d.shape[0] for d in non_max_suppression(p0, conf_thres=0.01, iou_thres=0.65)]
    cq = torch.quantile(conf.float(), torch.tensor(CONF_QUANTILES, device=conf.device))
    lq = torch.quantile(logit.float(), torch.tensor(LOGIT_QUANTILES, device=logit.device))
    print(f"\n=== {name} ===")
    print(f"  confidence: max={conf.max().item():.4g} mean={conf.mean().item():.4g} median={conf.median().item():.4g}")
    print(f"  confidence: p90={cq[1].item():.4g} p99={cq[2].item():.4g}")
    print(f"  raw logit : p1={lq[0].item():.4g} median={lq[1].item():.4g} p99={lq[2].item():.4g}")
    print(f"  NMS survivors/image: mean={np.mean(dets_per_img):.2f} max={max(dets_per_img)} nonzero={sum(d > 0 for d in dets_per_img)}/{len(dets_per_img)}")


def copy_bn_buffers(dst: nn.Module, src: nn.Module) -> None:
    dst_bns = [m for m in dst.modules() if isinstance(m, nn.BatchNorm2d)]
    src_bns = [m for m in src.modules() if isinstance(m, nn.BatchNorm2d)]
    assert len(dst_bns) == len(src_bns)
    with torch.no_grad():
        for d, s in zip(dst_bns, src_bns):
            d.running_mean.copy_(s.running_mean)
            d.running_var.copy_(s.running_var)
            d.num_batches_tracked.copy_(s.num_batches_tracked)


def recalibrate_bn(model: nn.Module, calib_batches: list[torch.Tensor], device: torch.device) -> None:
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats()
            m.momentum = None
    model.to(device).train()
    with torch.no_grad():
        for b in calib_batches:
            model(b.to(device))


def per_layer_bn_mismatch(model: nn.Module, batch: torch.Tensor, device: torch.device, tag: str) -> list[dict]:
    """Compare each BN layer's stored running stats vs the empirical stats of this batch's
    actual pre-BN activations at that layer (captured via forward hooks)."""
    model = copy.deepcopy(model).to(device).eval()
    captured = {}

    def make_hook(name):
        def hook(module, inp, out):
            x = inp[0].detach()
            mean = x.mean(dim=(0, 2, 3))
            var = x.var(dim=(0, 2, 3), unbiased=False)
            captured[name] = (mean.cpu(), var.cpu(), module.running_mean.detach().cpu(), module.running_var.detach().cpu())
        return hook

    handles = []
    idx = 0
    names = []
    for name, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            handles.append(m.register_forward_hook(make_hook(name)))
            names.append(name)
            idx += 1

    with torch.no_grad():
        model(batch.to(device))
    for h in handles:
        h.remove()

    rows = []
    for name in names:
        emp_mean, emp_var, stored_mean, stored_var = captured[name]
        mean_mismatch = (emp_mean - stored_mean).abs().mean().item()
        # relative variance mismatch (ratio-based, since raw var scales vary hugely across layers)
        var_ratio = (emp_var / stored_var.clamp_min(1e-12)).mean().item()
        var_mismatch_abs = (emp_var - stored_var).abs().mean().item()
        rows.append(
            {
                "layer": f"{tag}:{name}",
                "stored_mean_absmean": stored_mean.abs().mean().item(),
                "emp_mean_absmean": emp_mean.abs().mean().item(),
                "mean_mismatch": mean_mismatch,
                "stored_var_mean": stored_var.mean().item(),
                "emp_var_mean": emp_var.mean().item(),
                "var_mismatch_abs": var_mismatch_abs,
                "var_ratio_emp_over_stored": var_ratio,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--unlabeled-list", required=True)
    parser.add_argument("--n-images", type=int, default=32)
    parser.add_argument("--n-calib-batches", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    print(f"epoch={ckpt['epoch']} teacher_updates={ckpt['teacher_updates']} ema_updates={ckpt['updates']}")

    list_path = Path(args.unlabeled_list)
    test_batch = load_batch(list_path, args.n_images, args.imgsz, seed=0)
    print(f"test batch: {test_batch.shape}")

    teacher = ckpt["teacher"].float()
    student_proxy = ckpt["ema"].float()  # see module docstring re: raw student not checkpointed

    # A. Teacher weight + Teacher's own BN buffers
    report("A. Teacher weight + Teacher's own BN buffers", copy.deepcopy(teacher), test_batch, device)

    # B. Teacher weight + Student(proxy) BN buffers copied in
    b_model = copy.deepcopy(teacher)
    copy_bn_buffers(b_model, student_proxy)
    report("B. Teacher weight + Student(proxy) BN buffers copied in", b_model, test_batch, device)

    # C. Teacher weight + BN recalibrated on real current images
    c_model = copy.deepcopy(teacher)
    calib_batches = [load_batch(list_path, args.n_images, args.imgsz, seed=100 + i) for i in range(args.n_calib_batches)]
    recalibrate_bn(c_model, calib_batches, device)
    report(f"C. Teacher weight + BN recalibrated ({args.n_calib_batches} batches of real images)", c_model, test_batch, device)

    # D. Student(proxy) weight + Student(proxy)'s own BN buffers
    report("D. Student(proxy) weight + Student(proxy)'s own BN buffers", copy.deepcopy(student_proxy), test_batch, device)

    # Per-layer BN mismatch, both for Teacher and for Student(proxy), same batch.
    print("\n\n=== Per-layer BN mismatch (stored running stats vs this batch's empirical stats) ===")
    all_rows = []
    all_rows += per_layer_bn_mismatch(teacher, test_batch, device, "teacher")
    all_rows += per_layer_bn_mismatch(student_proxy, test_batch, device, "student")

    # Rank by mean_mismatch and by var_ratio deviation from 1.0 (both directions)
    by_mean = sorted(all_rows, key=lambda r: -r["mean_mismatch"])
    by_var = sorted(all_rows, key=lambda r: -abs(np.log(max(r["var_ratio_emp_over_stored"], 1e-12))))

    print("\n-- Top 10 layers by |mean mismatch| --")
    for r in by_mean[:10]:
        print(f"  {r['layer']:40s} stored_mean|.|={r['stored_mean_absmean']:.4g} emp_mean|.|={r['emp_mean_absmean']:.4g} mismatch={r['mean_mismatch']:.4g}")

    print("\n-- Top 10 layers by variance ratio deviation (emp_var / stored_var) --")
    for r in by_var[:10]:
        print(f"  {r['layer']:40s} stored_var={r['stored_var_mean']:.4g} emp_var={r['emp_var_mean']:.4g} ratio={r['var_ratio_emp_over_stored']:.4g}")


if __name__ == "__main__":
    main()
