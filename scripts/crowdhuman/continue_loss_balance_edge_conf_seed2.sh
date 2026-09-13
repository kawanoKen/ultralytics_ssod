#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/ultralytics_uv_cache}"
export YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-/tmp/ultralytics_yolo_config}"

DEVICES="0,1,2,3"
BATCH=128
SAVE_PERIOD=10
DIAG_INTERVAL=100

echo "=== resume seed2 / 10% ==="
uv run python scripts/voc_ssod/train_ssod.py --variant edge --arch yolov8n \
  --model runs/crowdhuman_ssod_10p_zero_pseudo_ablation/yolov8n_10p_loss_balance_edge_conf_seed2_valid/weights/last.pt \
  --data crowdhuman_10p.yaml --project runs/crowdhuman_ssod_10p_zero_pseudo_ablation \
  --name yolov8n_10p_loss_balance_edge_conf_seed2_valid --device "${DEVICES}" \
  --batch "${BATCH}" --batch_ssod "${BATCH}" --save_period "${SAVE_PERIOD}" \
  --diag-interval "${DIAG_INTERVAL}" --resume
echo "=== resume seed2 / 10% done (exit $?) ==="

echo "=== seed2 / 1% ==="
uv run python scripts/voc_ssod/train_ssod.py --variant edge --arch yolov8n \
  --model runs/crowdhuman_labeled_baseline_1p/yolov8n_crowdhuman_labeled/weights/best.pt \
  --data crowdhuman_1p.yaml --project runs/crowdhuman_ssod_1p_zero_pseudo_ablation \
  --name yolov8n_1p_loss_balance_edge_conf_seed2_valid --device "${DEVICES}" \
  --batch "${BATCH}" --batch_ssod "${BATCH}" --save_period "${SAVE_PERIOD}" \
  --diag-interval "${DIAG_INTERVAL}" --seed 2 --skip-zero-pseudo-cls-loss \
  --loss-balancing-mode ema_scale --loss-balance-beta 0.9
echo "=== seed2 / 1% done (exit $?) ==="
