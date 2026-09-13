"""Launch one SSOD run comparing pseudo-label filtering strategies.

Three configurations are compared:
  --variant baseline : filter reliable boxes by classification confidence only
  --variant dfl       : ALSO require whole-box DFL localization confidence (use_loc_conf) --
                        drops the entire box if any edge is uncertain
  --variant edge      : same reliable-box set as baseline (use_loc_conf=False), but within the
                        DFL loss for each reliable box, only edges with per-edge confidence
                        below edge_conf_threshold are excluded -- the other, more confident
                        edges of that same box still get trained (see EfficientTeacherLoss's
                        use_edge_conf / _bbox_loss_with_edge_mask)

Everything else (conf thresholds, ssod_weight, epochs, batch) is held fixed across variants so
only the filtering strategy differs.

burn_in_epochs=0: rather than re-running supervised burn-in inside this script, --model should
point at an already-converged supervised checkpoint (e.g. runs/voc_labeled_baseline/<arch>_voc07_labeled/
weights/best.pt) so training enters the pseudo-label phase from epoch 0.
"""

from __future__ import annotations

import argparse

from ultralytics import YOLO
from ultralytics.models.yolo.detect import SSODTrainer

CONF_THRESHOLD_HIGH = 0.5
CONF_THRESHOLD_LOW = 0.3
LOC_CONF_THRESHOLD = 0.6
EDGE_CONF_THRESHOLD = 0.6
BURN_IN_EPOCHS = 0
EPOCHS = 100
SSOD_WEIGHT = 0.5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=["baseline", "dfl", "edge"])
    parser.add_argument("--arch", required=True, help="short tag for output naming, e.g. yolov8n")
    parser.add_argument("--model", required=True, help="starting checkpoint, e.g. an already-converged best.pt")
    parser.add_argument("--data", default="VOC_ssod.yaml")
    parser.add_argument("--device", default="0,1")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--batch_ssod", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--edge-conf-mask-mode",
        default="selected",
        choices=["selected", "random"],
        help="per-edge DFL mask selection: DFL-selected positions or count-matched random positions",
    )
    parser.add_argument(
        "--edge-dfl-reweight",
        action="store_true",
        help="enable selector-independent edge-wise DFL dose-response weighting",
    )
    parser.add_argument("--edge-dfl-selector", choices=["oracle", "dfl"], default="dfl")
    parser.add_argument("--edge-dfl-weight", type=float, default=1.0)
    parser.add_argument("--no-edge-dfl-normalize", action="store_true")
    parser.add_argument("--oracle-edge-error-threshold", type=float, default=0.10)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--save_period", type=int, default=10)
    parser.add_argument("--project", default="runs/voc_ssod")
    parser.add_argument("--name", default=None, help="override the auto-generated run name")
    parser.add_argument(
        "--skip-zero-pseudo-cls-loss",
        action="store_true",
        help="skip the unsupervised classification loss for batches with zero adopted pseudo-labels "
        "(see EfficientTeacherLoss.skip_zero_pseudo_cls_loss)",
    )
    parser.add_argument(
        "--diag-interval",
        type=int,
        default=100,
        help="step interval for the heavier ssod_diagnostics.py logs (EMA norms, pseudo-label "
        "confidence distribution, BN mismatch probe)",
    )
    parser.add_argument(
        "--cls-loss-denom",
        default="target_score_sum",
        choices=["target_score_sum", "fixed", "ema"],
        help="unsupervised classification loss denominator: target_score_sum (default), a "
        "content-independent fixed reference (batch_size*anchors_per_image), or an EMA of "
        "sum_target_scores_unsup",
    )
    parser.add_argument("--ema-denom-beta", type=float, default=0.9)
    parser.add_argument("--ssod-weight", type=float, default=SSOD_WEIGHT)
    parser.add_argument(
        "--loss-balancing-mode",
        default="none",
        choices=["none", "ema_scale"],
        help="none (default) or ema_scale (Zoph et al.-style: scale L_unsup by EMA(L_sup)/EMA(L_unsup) "
        "before applying ssod_weight)",
    )
    parser.add_argument("--loss-balance-beta", type=float, default=0.9)
    parser.add_argument(
        "--supervised-only-control",
        action="store_true",
        help="compute/update-matched supervised-only control run: same optimizer, LR schedule, "
        "warmup, batch/DDP config and iteration count as a real SSOD run launched with the same "
        "--epochs/--batch/--batch_ssod/--device, but unlabeled images are never forwarded and no "
        "pseudo-label loss is computed (see SSODTrainer._do_train_supervised_only_control)",
    )
    args = parser.parse_args()

    use_loc_conf = args.variant == "dfl"
    use_edge_conf = args.variant == "edge"
    name = args.name or f"{args.arch}_voc_ssod_{args.variant}"

    model = YOLO(args.model)
    model.train(
        data=args.data,
        trainer=SSODTrainer,
        epochs=EPOCHS,
        burn_in_epochs=BURN_IN_EPOCHS,
        domain_adaptation=False,
        conf_threshold_high=CONF_THRESHOLD_HIGH,
        conf_threshold_low=CONF_THRESHOLD_LOW,
        use_loc_conf=use_loc_conf,
        loc_conf_threshold=LOC_CONF_THRESHOLD,
        use_edge_conf=use_edge_conf,
        edge_conf_threshold=EDGE_CONF_THRESHOLD,
        edge_conf_mask_mode=args.edge_conf_mask_mode,
        edge_dfl_reweight=args.edge_dfl_reweight,
        edge_dfl_selector=args.edge_dfl_selector,
        edge_dfl_weight=args.edge_dfl_weight,
        edge_dfl_normalize=not args.no_edge_dfl_normalize,
        oracle_edge_error_threshold=args.oracle_edge_error_threshold,
        ssod_weight=args.ssod_weight,
        skip_zero_pseudo_cls_loss=args.skip_zero_pseudo_cls_loss,
        cls_loss_denom=args.cls_loss_denom,
        ema_denom_beta=args.ema_denom_beta,
        loss_balancing_mode=args.loss_balancing_mode,
        loss_balance_beta=args.loss_balance_beta,
        supervised_only_control=args.supervised_only_control,
        diag_interval=args.diag_interval,
        imgsz=args.imgsz,
        batch=args.batch,
        batch_ssod=args.batch_ssod,
        seed=args.seed,
        resume=args.resume,
        device=args.device,
        save_period=args.save_period,
        project=args.project,
        name=name,
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
