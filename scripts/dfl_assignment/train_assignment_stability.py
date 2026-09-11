"""Run the frozen R1/R2 assignment-stability SSOD variants.

The starting model must be an existing supervised checkpoint. This script deliberately disables the existing
whole-box/edge DFL filters so the only experimental change is assignment stability.
"""

from __future__ import annotations

import argparse

from ultralytics import YOLO
from ultralytics.models.yolo.detect import SSODTrainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="r1", choices=("r1",))
    parser.add_argument("--perturbation", required=True, choices=("fixed", "width_matched"))
    parser.add_argument("--model", required=True, help="Existing supervised best.pt")
    parser.add_argument("--data", default="crowdhuman_5p.yaml")
    parser.add_argument("--device", default="0")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--batch-ssod", type=int, default=128)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--project", default="runs/dfl_assignment")
    parser.add_argument("--name", default=None)
    parser.add_argument("--val", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    name = args.name or f"yolov8n_crowdhuman_5p_{args.method}_{args.perturbation}"
    YOLO(args.model).train(
        trainer=SSODTrainer,
        data=args.data,
        assignment_stability_method=args.method,
        assignment_perturbation=args.perturbation,
        use_loc_conf=False,
        use_edge_conf=False,
        burn_in_epochs=0,
        domain_adaptation=False,
        conf_threshold_high=0.5,
        conf_threshold_low=0.3,
        ssod_weight=0.5,
        epochs=args.epochs,
        batch=args.batch,
        batch_ssod=args.batch_ssod,
        imgsz=args.imgsz,
        workers=args.workers,
        fraction=args.fraction,
        device=args.device,
        val=args.val,
        save=args.save,
        plots=False,
        pseudo_label_plots=False,
        seed=0,
        deterministic=True,
        save_period=10,
        project=args.project,
        name=name,
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
