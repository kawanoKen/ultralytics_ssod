"""Step 1-2: single-GPU CrowdHuman 1% run with spike_diag enabled.

Reproduces the labeled-only-baseline SSOD recipe that originally showed ssod/cls_loss spikes
(runs/crowdhuman_ssod_1p/yolov8n_voc_ssod_baseline), on a single GPU with workers=0 so the
dataloader RNG/order is fully reproducible for later state snapshot/restore. Stops as soon as the
first spike (ssod/cls_loss > --threshold) is captured, unless --no-stop is passed.
"""

from __future__ import annotations

import argparse

from ultralytics import YOLO
from ultralytics.models.yolo.detect import SSODTrainer
from ultralytics.utils.loss_spike_lab import SpikeCaptured

CONF_THRESHOLD_HIGH = 0.5
CONF_THRESHOLD_LOW = 0.3
SSOD_WEIGHT = 0.5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="runs/crowdhuman_labeled_baseline_1p/yolov8n_crowdhuman_labeled/weights/best.pt")
    parser.add_argument("--data", default="crowdhuman_1p.yaml")
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--batch_ssod", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=10.0)
    parser.add_argument("--project", default="runs/spike_diag_1p")
    parser.add_argument("--name", default="reproduce")
    parser.add_argument("--no-stop", action="store_true", help="keep training after the first spike capture")
    args = parser.parse_args()

    model = YOLO(args.model)
    try:
        model.train(
            data=args.data,
            trainer=SSODTrainer,
            epochs=args.epochs,
            burn_in_epochs=0,
            domain_adaptation=False,
            conf_threshold_high=CONF_THRESHOLD_HIGH,
            conf_threshold_low=CONF_THRESHOLD_LOW,
            use_loc_conf=False,
            use_edge_conf=False,
            ssod_weight=SSOD_WEIGHT,
            imgsz=640,
            batch=args.batch,
            batch_ssod=args.batch_ssod,
            device=args.device,
            workers=0,  # required for deterministic, restorable dataloader state
            seed=0,
            deterministic=True,
            save_period=10,
            project=args.project,
            name=args.name,
            exist_ok=True,
            spike_diag_enabled=True,
            spike_diag_threshold=args.threshold,
            spike_diag_stop_after_capture=not args.no_stop,
        )
    except SpikeCaptured as e:
        print(f"SPIKE_CAPTURED: {e}")


if __name__ == "__main__":
    main()
