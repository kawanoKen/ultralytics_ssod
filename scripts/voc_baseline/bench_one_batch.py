"""Run a single-GPU, 2-epoch training probe at one batch size and report steady-state throughput.

Called once per (model, batch) combination by bench_batch_size.sh, each in its own process so
an OOM at one batch size can't take down the rest of the sweep. Epoch 1 includes cache warm-up
(dataset caching to RAM) and CUDA/cuDNN warm-up, so only epoch 2's wall-clock time is used to
compute images/sec.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import torch

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    timings: dict[str, float | list[float]] = {}

    def on_epoch_start(trainer) -> None:
        timings["start"] = time.time()

    def on_epoch_end(trainer) -> None:
        timings.setdefault("epochs", []).append(time.time() - timings["start"])
        # number of training images actually seen this epoch (post-'fraction', pre-drop_last)
        timings["num_images"] = len(trainer.train_loader.dataset)

    model = YOLO(args.model)
    model.add_callback("on_train_epoch_start", on_epoch_start)
    model.add_callback("on_train_epoch_end", on_epoch_end)

    result = {"model": args.model, "batch": args.batch, "status": "ok"}
    try:
        model.train(
            data=args.data,
            epochs=2,
            imgsz=args.imgsz,
            batch=args.batch,
            device=int(args.device) if args.device.isdigit() else args.device,
            fraction=1.0,
            cache="ram",
            val=False,
            plots=False,
            workers=args.workers,
            project="runs/bench_batch",
            name=f"{args.model.replace('.pt', '')}_bs{args.batch}",
            exist_ok=True,
            verbose=False,
        )
        epoch2_time = timings["epochs"][1]
        num_images = timings["num_images"]
        result.update(
            epoch2_seconds=epoch2_time,
            num_images=num_images,
            images_per_sec=num_images / epoch2_time,
        )
    except torch.cuda.OutOfMemoryError:
        result["status"] = "OOM"
    except Exception as e:  # noqa: BLE001 - surface any other failure to the sweep table
        result["status"] = f"error: {e}"

    print("RESULT_JSON " + json.dumps(result))


if __name__ == "__main__":
    sys.exit(main())
