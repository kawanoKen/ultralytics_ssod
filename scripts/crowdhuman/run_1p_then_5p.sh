#!/usr/bin/env bash
# Runs the full 6-experiment CrowdHuman pipeline for 1% labeled, then 5% labeled.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

bash scripts/crowdhuman/run_full_pipeline.sh 1p
bash scripts/crowdhuman/run_full_pipeline.sh 5p

echo "=== 1p and 5p pipelines both finished. ==="
