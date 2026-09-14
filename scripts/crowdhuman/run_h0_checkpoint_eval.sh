#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/ultralytics_uv_cache}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
worker="${1:?worker 0..3}"
device="${2:?device}"
mkdir -p 修論/H0/checkpoint_eval
tasks=()
for noise in low high; do
  for mode in on off; do
    for seed in 0 1; do tasks+=("${noise},${mode},${seed}"); done
  done
done
for i in "${!tasks[@]}"; do
  (( i % 4 == worker )) || continue
  IFS=, read -r noise mode seed <<< "${tasks[$i]}"
  run="runs/crowdhuman_h0_boundary_noise/yolov8n_full_h0_${noise}_dfl_${mode}_seed${seed}"
  ionice -c3 nice -n 19 uv run python scripts/crowdhuman/evaluate_h0_checkpoints.py --checkpoint "${run}/weights/best.pt" --out "修論/H0/checkpoint_eval/${noise}_${mode}_seed${seed}.csv" --device "cuda:${device}" --batch-size 1
done
