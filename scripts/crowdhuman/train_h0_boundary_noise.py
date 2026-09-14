"""Run one H0 artificial-boundary-noise fully supervised experiment."""

from __future__ import annotations

import argparse

from ultralytics import YOLO


NOISE_FRACTIONS = {"low": 0.05, "high": 0.20}  # fixed from prior pseudo-edge-error statistics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise", required=True, choices=NOISE_FRACTIONS)
    parser.add_argument(
        "--dfl",
        required=True,
        choices=["on", "off", "clean"],
        help="on=noisy DFL target, off=mask corrupted DFL edge, clean=clean DFL target with noisy IoU target",
    )
    parser.add_argument("--seed", required=True, type=int, choices=[0, 1])
    parser.add_argument("--device", default="0,1,2,3")
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--project", default="runs/crowdhuman_h0_boundary_noise")
    parser.add_argument("--name", default=None)
    args = parser.parse_args()

    name = args.name or f"yolov8n_full_h0_{args.noise}_dfl_{args.dfl}_seed{args.seed}"
    model = YOLO("yolov8n.pt")
    model.train(
        data="crowdhuman_full.yaml",
        epochs=args.epochs,
        imgsz=640,
        batch=args.batch,
        device=args.device,
        seed=args.seed,
        save_period=10,
        project=args.project,
        name=name,
        exist_ok=True,
        h0_boundary_noise=True,
        h0_noise_fraction=NOISE_FRACTIONS[args.noise],
        h0_dfl_mode=args.dfl,
        h0_corruption_seed=20260914,
    )


if __name__ == "__main__":
    main()
