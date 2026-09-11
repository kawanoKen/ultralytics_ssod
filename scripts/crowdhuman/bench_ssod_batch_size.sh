#!/usr/bin/env bash
# Single-GPU OOM/throughput sweep for CrowdHuman's SSOD pseudo-label phase (heavier
# than plain supervised: teacher inference + student forward/backward on both
# labeled and unlabeled batches per step).
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

DATA="crowdhuman_10p.yaml"
DEVICE="0"
BATCHES=(16 32 64)
MODELS=(yolov8n.pt yolo11n.pt)

LOG_DIR="scripts/crowdhuman/bench_logs"
mkdir -p "${LOG_DIR}"
SUMMARY="${LOG_DIR}/ssod_summary.tsv"
echo -e "model\tbatch\tbatch_ssod\tstatus\tepoch2_seconds" > "${SUMMARY}"

for model in "${MODELS[@]}"; do
  for batch in "${BATCHES[@]}"; do
    echo "==> ${model} batch=${batch} batch_ssod=${batch}"
    log="${LOG_DIR}/ssod_${model%.pt}_bs${batch}.log"
    if uv run python scripts/voc_ssod/bench_one_ssod_batch.py \
        --model "${model}" --data "${DATA}" --batch "${batch}" --batch_ssod "${batch}" --device "${DEVICE}" \
        > "${log}" 2>&1; then
      line=$(grep -o 'RESULT_JSON .*' "${log}" | tail -1 | sed 's/RESULT_JSON //')
    else
      line='{"model": "'"${model}"'", "batch": '"${batch}"', "batch_ssod": '"${batch}"', "status": "crashed"}'
    fi
    python3 - "$line" <<'PY' >> "${SUMMARY}"
import json, sys
d = json.loads(sys.argv[1])
print(f"{d['model']}\t{d['batch']}\t{d['batch_ssod']}\t{d['status']}\t{d.get('epoch2_seconds', '')}")
PY
  done
done

echo
echo "=== Summary (${SUMMARY}) ==="
column -t "${SUMMARY}"
