#!/usr/bin/env bash
# Clean-GT signed-edge-error and area-ratio evaluation for the directional H0 study.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/ultralytics_uv_cache}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

device="${1:-0}"
out_dir="修論/H0/directional_noise_eval"
mkdir -p "${out_dir}"
# The a20/out50 reference is the existing High symmetric run, whose legacy name differs.
for dfl in on clean; do
  seed=0
  for spec in 'a20_out100' 'a20_out0' 'a20_out80' 'a40_out50'; do
    run="runs/crowdhuman_h0_boundary_noise/yolov8n_full_h0_${spec}_dfl_${dfl}_seed${seed}"
    uv run python scripts/crowdhuman/evaluate_h0_checkpoints.py \
      --checkpoint "${run}/weights/best.pt" --out "${out_dir}/${spec}_dfl_${dfl}_seed${seed}.csv" \
      --device "cuda:${device}" --batch-size 1
  done
done
for dfl in on clean; do
  seed=0
  uv run python scripts/crowdhuman/evaluate_h0_checkpoints.py \
    --checkpoint "runs/crowdhuman_h0_boundary_noise/yolov8n_full_h0_high_dfl_${dfl}_seed${seed}/weights/best.pt" \
    --out "${out_dir}/a20_out50_dfl_${dfl}_seed${seed}.csv" --device "cuda:${device}" --batch-size 1
done
