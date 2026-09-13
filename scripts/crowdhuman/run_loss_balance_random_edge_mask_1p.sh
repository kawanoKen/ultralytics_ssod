#!/usr/bin/env bash
# CrowdHuman 1% count-matched random edge-mask ablation.
#
# The DFL-selected mask is used only to obtain the per-batch count for each
# edge (left, top, right, bottom).  The random variant then samples the same
# count from eligible reliable foreground anchors independently of edge DFL
# confidence.  All other SSOD settings match the existing loss_balance +
# DFL edge-confidence runs.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/ultralytics_uv_cache}"
export YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-/tmp/ultralytics_yolo_config}"

DEVICES="0,1,2,3"
BATCH=128
SAVE_PERIOD=10
DIAG_INTERVAL=100
MODEL_1P="runs/crowdhuman_labeled_baseline_1p/yolov8n_crowdhuman_labeled/weights/best.pt"
PROJECT_1P="runs/crowdhuman_ssod_1p_zero_pseudo_ablation"

for seed in 0 1 2; do
  name="yolov8n_1p_loss_balance_random_edge_mask_seed${seed}"
  echo "=== ${name} ==="
  uv run python scripts/voc_ssod/train_ssod.py --variant edge --arch yolov8n \
    --model "${MODEL_1P}" --data crowdhuman_1p.yaml --project "${PROJECT_1P}" --name "${name}" \
    --device "${DEVICES}" --batch "${BATCH}" --batch_ssod "${BATCH}" \
    --save_period "${SAVE_PERIOD}" --diag-interval "${DIAG_INTERVAL}" \
    --seed "${seed}" --edge-conf-mask-mode random \
    --skip-zero-pseudo-cls-loss --loss-balancing-mode ema_scale \
    --loss-balance-beta 0.9
  echo "=== ${name} done (exit $?) ==="
done

echo "=== CrowdHuman 1% random count-matched edge-mask runs finished. ==="
