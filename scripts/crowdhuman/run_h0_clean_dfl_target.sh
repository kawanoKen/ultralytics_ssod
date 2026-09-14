#!/usr/bin/env bash
# H0 follow-up: retain the noisy IoU target but restore the corrupted DFL edge to clean GT.
# Existing H0 DFL-ON runs are the paired noisy-DFL controls; this script launches only four new runs.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/ultralytics_uv_cache}"
export YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-/tmp/ultralytics_yolo_config}"
mkdir -p scripts/crowdhuman/logs

devices="${1:-0,1,2,3}"
for noise in low high; do
  for seed in 0 1; do
    uv run python scripts/crowdhuman/train_h0_boundary_noise.py \
      --noise "${noise}" --dfl clean --seed "${seed}" --device "${devices}"
  done
done
