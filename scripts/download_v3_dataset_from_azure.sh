#!/usr/bin/env bash
# Download the normalized v3 dataset onto RunPod.

set -Eeuo pipefail

if ! command -v azcopy >/dev/null 2>&1; then
    echo "azcopy is required." >&2
    exit 1
fi
if [[ -z "${AZURE_V3_BASE_URL:-}" || -z "${AZURE_V3_SAS:-}" ]]; then
    echo "Set AZURE_V3_BASE_URL and AZURE_V3_SAS first." >&2
    exit 1
fi

DATASET_ROOT="${1:-/workspace/datasets/page_layout_v3}"
BASE="${AZURE_V3_BASE_URL%/}"
mkdir -p "${DATASET_ROOT}/annotations/coco"

for split in train valid test; do
    mkdir -p "${DATASET_ROOT}/images/${split}" "${DATASET_ROOT}/labels/${split}"
    azcopy copy \
        "${BASE}/images/${split}/*${AZURE_V3_SAS}" \
        "${DATASET_ROOT}/images/${split}" \
        --recursive=true --check-md5=FailIfDifferent
    azcopy copy \
        "${BASE}/labels/${split}/*${AZURE_V3_SAS}" \
        "${DATASET_ROOT}/labels/${split}" \
        --recursive=true --check-md5=FailIfDifferent
    azcopy copy \
        "${BASE}/annotations/coco/${split}.json${AZURE_V3_SAS}" \
        "${DATASET_ROOT}/annotations/coco/${split}.json" \
        --check-md5=FailIfDifferent
done

echo "V3 dataset downloaded to ${DATASET_ROOT}"

