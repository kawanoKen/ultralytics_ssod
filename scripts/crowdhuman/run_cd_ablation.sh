#!/usr/bin/env bash
# C (EMA denominator) + D (EMA loss-scale balancing) ablation, 1% and 10% CrowdHuman.
# Second machine's half of run_bcd_ablation.sh; the other half (B: fixed lower ssod_weight)
# runs on the first machine -- see scripts/crowdhuman/run_b_weight_ablation.sh.
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

# C: EMA denominator
run "${MODEL_10P}" "${PROJECT_10P}" crowdhuman_10p.yaml yolov8n_10p_ema_denom --cls-loss-denom ema
run "${MODEL_1P}"  "${PROJECT_1P}"  crowdhuman_1p.yaml  yolov8n_1p_ema_denom  --cls-loss-denom ema

# D: EMA loss-scale balancing
run "${MODEL_10P}" "${PROJECT_10P}" crowdhuman_10p.yaml yolov8n_10p_loss_balance --loss-balancing-mode ema_scale
run "${MODEL_1P}"  "${PROJECT_1P}"  crowdhuman_1p.yaml  yolov8n_1p_loss_balance  --loss-balancing-mode ema_scale

echo "=== C/D ablation finished. ==="
