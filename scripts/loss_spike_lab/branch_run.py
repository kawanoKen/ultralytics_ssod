"""Step 3: one counterfactual branch (NORMAL / SKIP-U / CLIP-U), run from scratch with the SAME
seed/data order as reproduce_1p_spike.py.

Rationale: with seed=0, workers=0, and deterministic=True, a from-scratch rerun of the exact same
recipe should retrace the identical trajectory (dataloader order, augmentation, model init, weight
updates) up to the intervention iteration -- so branches only need to differ in how the loss is
combined at --intervention-iter, rather than requiring an explicit state snapshot/restore fork.
This assumption is checked separately (two NORMAL runs should match at the spike iteration to
within floating point).

--intervention-iter should be the `iteration` value (global ni) from the reproduce run's
spike_diag.jsonl record where ssod_cls_loss first exceeded the trigger threshold.
"""

from __future__ import annotations

import argparse

from ultralytics import YOLO
from ultralytics.models.yolo.detect import SSODTrainer

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
    parser.add_argument("--intervention", required=True, choices=["normal", "skip_u", "clip_u"])
    parser.add_argument("--intervention-iter", type=int, required=True)
    parser.add_argument("--g-ref", type=float, default=None, help="required for clip_u: pre-declared ||g_u|| cap")
    parser.add_argument("--spike-threshold", type=float, default=10.0, help="kept identical to the reproduce run so logging/capture stays comparable")
    parser.add_argument("--project", default="runs/spike_diag_1p")
    parser.add_argument("--name", required=True)
    args = parser.parse_args()

    if args.intervention == "clip_u" and args.g_ref is None:
        parser.error("--g-ref is required for --intervention clip_u")

    model = YOLO(args.model)
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
        workers=0,
        seed=0,
        deterministic=True,
        save_period=10,
        project=args.project,
        name=args.name,
        exist_ok=True,
        spike_diag_enabled=True,
        spike_diag_threshold=args.spike_threshold,
        spike_diag_stop_after_capture=False,  # let it run to completion for this branch
        spike_diag_intervention=args.intervention,
        spike_diag_intervention_iter=args.intervention_iter,
        spike_diag_g_ref=args.g_ref,
    )


if __name__ == "__main__":
    main()
