#!/usr/bin/env bash
# Quick single-GPU throughput/OOM sweep for CrowdHuman (denser scenes than VOC --
# ~23 persons/image -- so mosaic-augmented batches can use more VRAM per image).
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

DATA="crowdhuman_10p.yaml"
DEVICE="0"
BATCHES=(16 32 64 128)
MODELS=(yolov8n.pt yolo11n.pt)

LOG_DIR="scripts/crowdhuman/bench_logs"
mkdir -p "${LOG_DIR}"
SUMMARY="${LOG_DIR}/summary.tsv"
echo -e "model\tbatch\tstatus\timages_per_sec\tepoch2_seconds" > "${SUMMARY}"

for model in "${MODELS[@]}"; do
  for batch in "${BATCHES[@]}"; do
    echo "==> ${model} batch=${batch}"
    log="${LOG_DIR}/${model%.pt}_bs${batch}.log"
    if uv run python scripts/voc_baseline/bench_one_batch.py \
        --model "${model}" --data "${DATA}" --batch "${batch}" --device "${DEVICE}" \
        > "${log}" 2>&1; then
      line=$(grep -o 'RESULT_JSON .*' "${log}" | tail -1 | sed 's/RESULT_JSON //')
    else
      line='{"model": "'"${model}"'", "batch": '"${batch}"', "status": "crashed"}'
    fi
    python3 - "$line" <<'PY' >> "${SUMMARY}"
import json, sys
d = json.loads(sys.argv[1])
print(f"{d['model']}\t{d['batch']}\t{d['status']}\t{d.get('images_per_sec', '')}\t{d.get('epoch2_seconds', '')}")
PY
  done
done

echo
echo "=== Summary (${SUMMARY}) ==="
column -t "${SUMMARY}"
