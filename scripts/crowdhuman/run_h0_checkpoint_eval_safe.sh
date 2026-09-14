#!/usr/bin/env bash
# Conservative, inference-only H0 evaluation: one GPU, one image/batch, low CPU/I/O priority.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/ultralytics_uv_cache}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
mkdir -p 修論/H0/checkpoint_eval

for noise in low high; do
  for mode in on off; do
    for seed in 0 1; do
      run="runs/crowdhuman_h0_boundary_noise/yolov8n_full_h0_${noise}_dfl_${mode}_seed${seed}"
      ionice -c3 nice -n 19 uv run python scripts/crowdhuman/evaluate_h0_checkpoints.py \
        --checkpoint "${run}/weights/best.pt" \
        --out "修論/H0/checkpoint_eval/${noise}_${mode}_seed${seed}.csv" \
        --device cuda:0 --batch-size 1
    done
  done
done
