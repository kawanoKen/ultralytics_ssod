#!/usr/bin/env bash
# Loss-balance + DFL per-edge confidence masking on CrowdHuman.
#
# Compared with the existing loss_balance runs, this changes only the SSOD
# localization branch:
#   - variant=edge: keep the classification-confidence pseudo-label set
#   - use per-edge DFL confidence to mask low-confidence edges in the DFL loss
#   - edge_conf_threshold is 0.6 (the train_ssod.py default)
#
# The loss-balance settings match the previous runs: ssod_weight=0.5,
# skip_zero_pseudo_cls_loss=True, loss_balancing_mode=ema_scale, beta=0.9,
# batch=128, batch_ssod=128, seed=0, and 100 epochs.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
# Keep the launcher usable on managed/read-only environments as well.
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/ultralytics_uv_cache}"
export YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-/tmp/ultralytics_yolo_config}"

DEVICES="0,1,2,3"
BATCH=128
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
  uv run python scripts/voc_ssod/train_ssod.py --variant edge --arch yolov8n \
    --model "${model}" --data "${data}" --project "${project}" --name "${name}" \
    --device "${DEVICES}" --batch "${BATCH}" --batch_ssod "${BATCH}" \
    --save_period "${SAVE_PERIOD}" --diag-interval "${DIAG_INTERVAL}" \
    --skip-zero-pseudo-cls-loss --loss-balancing-mode ema_scale \
    --loss-balance-beta 0.9 "$@"
  echo "=== ${name} done (exit $?) ==="
}

run "${MODEL_10P}" "${PROJECT_10P}" crowdhuman_10p.yaml yolov8n_10p_loss_balance_edge_conf
run "${MODEL_1P}"  "${PROJECT_1P}"  crowdhuman_1p.yaml  yolov8n_1p_loss_balance_edge_conf

echo "=== Loss-balance + DFL edge-confidence runs finished. ==="
