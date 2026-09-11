#!/usr/bin/env bash
# B: fixed lower ssod_weight ablation (0.1, 0.25), 1% and 10% CrowdHuman.
# This machine's half of run_bcd_ablation.sh; the other half (C: ema denom, D: loss balancing)
# runs on a second machine -- see scripts/crowdhuman/run_cd_ablation.sh.
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

run "${MODEL_10P}" "${PROJECT_10P}" crowdhuman_10p.yaml yolov8n_10p_weight010 --ssod-weight 0.1
run "${MODEL_1P}"  "${PROJECT_1P}"  crowdhuman_1p.yaml  yolov8n_1p_weight010  --ssod-weight 0.1
run "${MODEL_10P}" "${PROJECT_10P}" crowdhuman_10p.yaml yolov8n_10p_weight025 --ssod-weight 0.25
run "${MODEL_1P}"  "${PROJECT_1P}"  crowdhuman_1p.yaml  yolov8n_1p_weight025  --ssod-weight 0.25

echo "=== B weight ablation finished. ==="
