#!/usr/bin/env bash
# Inference-only clean-GT/AP75 evaluation for the four new H0 Clean-DFL checkpoints.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/ultralytics_uv_cache}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

device="${1:-0}"
mkdir -p 修論/H0/checkpoint_eval
for noise in low high; do
  for seed in 0 1; do
    run="runs/crowdhuman_h0_boundary_noise/yolov8n_full_h0_${noise}_dfl_clean_seed${seed}"
    uv run python scripts/crowdhuman/evaluate_h0_checkpoints.py \
      --checkpoint "${run}/weights/best.pt" \
      --out "修論/H0/checkpoint_eval/${noise}_clean_seed${seed}.csv" \
      --device "cuda:${device}" --batch-size 1
  done
done
