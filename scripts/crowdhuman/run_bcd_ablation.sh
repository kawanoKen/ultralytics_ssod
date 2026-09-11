#!/usr/bin/env bash
# B/C/D normalization & loss-balancing ablation, 1% and 10% CrowdHuman.
# Does NOT use `set -e` -- an unexpected crash in one run must not abort the rest of the queue
# (see the earlier zero_pseudo_ablation.sh incident where an *expected* NaN crash killed the
# whole chain). Everything else (init checkpoint, seed, LR/update count, thresholds, ssod_weight
# default, skip_zero_pseudo_cls_loss, gradient clipping) is held fixed except the one variable
# under test in each run.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

DEVICES="0,1,2,3"
BATCH=256
SAVE_PERIOD=10
DIAG_INTERVAL=100
MODEL_1P="runs/crowdhuman_labeled_baseline_1p/yolov8n_crowdhuman_labeled/weights/best.pt"
MODEL_10P="runs/crowdhuman_labeled_baseline/yolov8n_crowdhuman_labeled/weights/best.pt"
PROJECT_1P="runs/crowdhuman_ssod_1p_zero_pseudo_ablation"
PROJECT_10P="runs/crowdhuman_ssod_10p_zero_pseudo_ablation"

run() {
  local model=$1 project=$2 data=$3 name=$4
  shift 4
  echo "=== ${name} ==="
  uv run python scripts/voc_ssod/train_ssod.py --variant baseline --arch yolov8n \
    --model "${model}" --data "${data}" --project "${project}" --name "${name}" \
    --device "${DEVICES}" --batch "${BATCH}" --batch_ssod "${BATCH}" --save_period "${SAVE_PERIOD}" \
    --diag-interval "${DIAG_INTERVAL}" --skip-zero-pseudo-cls-loss "$@"
  echo "=== ${name} done (exit $?) ==="
}

# B: fixed lower ssod_weight (0.5 == current A, already have it; test 0.1 and 0.25)
run "${MODEL_10P}" "${PROJECT_10P}" crowdhuman_10p.yaml yolov8n_10p_weight010 --ssod-weight 0.1
run "${MODEL_1P}"  "${PROJECT_1P}"  crowdhuman_1p.yaml  yolov8n_1p_weight010  --ssod-weight 0.1
run "${MODEL_10P}" "${PROJECT_10P}" crowdhuman_10p.yaml yolov8n_10p_weight025 --ssod-weight 0.25
run "${MODEL_1P}"  "${PROJECT_1P}"  crowdhuman_1p.yaml  yolov8n_1p_weight025  --ssod-weight 0.25

# C: EMA denominator
run "${MODEL_10P}" "${PROJECT_10P}" crowdhuman_10p.yaml yolov8n_10p_ema_denom --cls-loss-denom ema
run "${MODEL_1P}"  "${PROJECT_1P}"  crowdhuman_1p.yaml  yolov8n_1p_ema_denom  --cls-loss-denom ema

# D: EMA loss-scale balancing
run "${MODEL_10P}" "${PROJECT_10P}" crowdhuman_10p.yaml yolov8n_10p_loss_balance --loss-balancing-mode ema_scale
run "${MODEL_1P}"  "${PROJECT_1P}"  crowdhuman_1p.yaml  yolov8n_1p_loss_balance  --loss-balancing-mode ema_scale

echo "=== All B/C/D ablation runs finished. ==="
