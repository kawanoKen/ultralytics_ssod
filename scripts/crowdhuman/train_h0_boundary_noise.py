"""Run one H0 artificial-boundary-noise fully supervised experiment."""

from __future__ import annotations

import argparse

from ultralytics import YOLO


NOISE_FRACTIONS = {"low": 0.05, "high": 0.20}  # fixed from prior pseudo-edge-error statistics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise", choices=NOISE_FRACTIONS, help="Legacy fixed alpha alias: low=0.05, high=0.20")
    parser.add_argument("--alpha", type=float, help="Relative noisy-edge scale; use for directional H0 variants")
    parser.add_argument("--outward-prob", type=float, default=0.5, help="P(outward); l/t decrease and r/b increase")
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
    if (args.noise is None) == (args.alpha is None):
        parser.error("specify exactly one of --noise or --alpha")
    if args.alpha is not None and args.alpha <= 0:
        parser.error("--alpha must be positive")
    if not 0.0 <= args.outward_prob <= 1.0:
        parser.error("--outward-prob must be in [0, 1]")

    alpha = NOISE_FRACTIONS[args.noise] if args.noise is not None else args.alpha
    if args.name:
        name = args.name
    elif args.noise is not None and args.outward_prob == 0.5:
        name = f"yolov8n_full_h0_{args.noise}_dfl_{args.dfl}_seed{args.seed}"
    else:
        alpha_tag = f"a{alpha * 100:g}".replace(".", "p")
        direction_tag = f"out{args.outward_prob * 100:g}".replace(".", "p")
        name = f"yolov8n_full_h0_{alpha_tag}_{direction_tag}_dfl_{args.dfl}_seed{args.seed}"
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
        h0_noise_fraction=alpha,
        h0_outward_probability=args.outward_prob,
        h0_dfl_mode=args.dfl,
        h0_corruption_seed=20260914,
    )


if __name__ == "__main__":
    main()
