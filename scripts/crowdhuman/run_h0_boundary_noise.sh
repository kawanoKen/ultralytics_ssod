#!/usr/bin/env bash
# H0: 100% labeled CrowdHuman, one noisy edge/GT, paired DFL ON/OFF.
# The fixed scales are Low=5% and High=20% of the affected box dimension.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/ultralytics_uv_cache}"
export YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-/tmp/ultralytics_yolo_config}"

for noise in low high; do
  for dfl in on off; do
    for seed in 0 1; do
      uv run python scripts/crowdhuman/train_h0_boundary_noise.py \
        --noise "${noise}" --dfl "${dfl}" --seed "${seed}" --device "${1:-0,1,2,3}"
    done
  done
done
