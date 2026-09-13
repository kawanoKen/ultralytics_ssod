#!/usr/bin/env bash
# Valid additional seeds for loss-balance + DFL per-edge confidence masking.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

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
  local model=$1 project=$2 data=$3 name=$4 seed=$5
  echo "=== ${name} (seed=${seed}) ==="
  uv run python scripts/voc_ssod/train_ssod.py --variant edge --arch yolov8n \
    --model "${model}" --data "${data}" --project "${project}" --name "${name}" \
    --device "${DEVICES}" --batch "${BATCH}" --batch_ssod "${BATCH}" \
    --save_period "${SAVE_PERIOD}" --diag-interval "${DIAG_INTERVAL}" \
    --seed "${seed}" --skip-zero-pseudo-cls-loss --loss-balancing-mode ema_scale \
    --loss-balance-beta 0.9
  echo "=== ${name} done (exit $?) ==="
}

for seed in 1 2; do
  run "${MODEL_10P}" "${PROJECT_10P}" crowdhuman_10p.yaml "yolov8n_10p_loss_balance_edge_conf_seed${seed}_valid" "${seed}"
  run "${MODEL_1P}"  "${PROJECT_1P}"  crowdhuman_1p.yaml  "yolov8n_1p_loss_balance_edge_conf_seed${seed}_valid"  "${seed}"
done

echo "=== Valid additional loss-balance + DFL edge-confidence seeds finished. ==="
