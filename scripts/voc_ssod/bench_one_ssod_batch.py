"""Single-GPU OOM/throughput probe for the SSOD pseudo-label phase (labeled + unlabeled forward/backward
per step, unlike the plain-supervised bench in scripts/voc_baseline/). burn_in_epochs=0 so the
pseudo-label loop starts immediately, matching the steady-state cost we actually care about.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import torch

from ultralytics import YOLO
from ultralytics.models.yolo.detect import SSODTrainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--batch", type=int, required=True, help="labeled batch size")
    parser.add_argument("--batch_ssod", type=int, required=True, help="unlabeled batch size")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    timings: dict = {}

    def on_epoch_start(trainer) -> None:
        timings["start"] = time.time()

    def on_epoch_end(trainer) -> None:
        timings.setdefault("epochs", []).append(time.time() - timings["start"])

    model = YOLO(args.model)
    model.add_callback("on_train_epoch_start", on_epoch_start)
    model.add_callback("on_train_epoch_end", on_epoch_end)

    result = {"model": args.model, "batch": args.batch, "batch_ssod": args.batch_ssod, "status": "ok"}
    try:
        model.train(
            data=args.data,
            trainer=SSODTrainer,
            epochs=2,
            burn_in_epochs=0,
            domain_adaptation=False,
            imgsz=args.imgsz,
            batch=args.batch,
            batch_ssod=args.batch_ssod,
            device=int(args.device) if args.device.isdigit() else args.device,
            fraction=1.0,
            val=False,
            plots=False,
            pseudo_label_plots=False,
            workers=args.workers,
            project="runs/bench_ssod_batch",
            name=f"{args.model.replace('.pt', '')}_b{args.batch}_bu{args.batch_ssod}",
            exist_ok=True,
            verbose=False,
        )
        epoch2_time = timings["epochs"][1]
        result.update(epoch2_seconds=epoch2_time)
    except torch.cuda.OutOfMemoryError:
        result["status"] = "OOM"
    except Exception as e:  # noqa: BLE001
        result["status"] = f"error: {e}"

    print("RESULT_JSON " + json.dumps(result))


if __name__ == "__main__":
    sys.exit(main())
