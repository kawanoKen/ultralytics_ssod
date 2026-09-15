#!/usr/bin/env bash
# Directional H0 study. Existing a20/out50 (the High symmetric runs) are reused, not rerun.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/ultralytics_uv_cache}"
export YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-/tmp/ultralytics_yolo_config}"
mkdir -p scripts/crowdhuman/logs

devices="${1:-0,1,2,3}"
# tag alpha outward_probability.  a20/out50 already exists as the High symmetric paired baseline.
for spec in 'a20 0.20 1.00' 'a20 0.20 0.00' 'a20 0.20 0.80' 'a40 0.40 0.50'; do
  read -r tag alpha outward_prob <<<"${spec}"
  for dfl in on clean; do
    for seed in 0 1; do
      name="yolov8n_full_h0_${tag}_out$(awk -v p="${outward_prob}" 'BEGIN {printf "%g", p * 100}')_dfl_${dfl}_seed${seed}"
      uv run python scripts/crowdhuman/train_h0_boundary_noise.py \
        --alpha "${alpha}" --outward-prob "${outward_prob}" --dfl "${dfl}" --seed "${seed}" \
        --device "${devices}" --name "${name}"
    done
  done
done
