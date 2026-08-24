#!/usr/bin/env bash
# Upload the normalized contents of the two Roboflow v3 exports without
# duplicating their byte-identical image files in Azure.

set -Eeuo pipefail

if ! command -v azcopy >/dev/null 2>&1; then
    echo "azcopy is required." >&2
    exit 1
fi
if [[ -z "${AZURE_V3_BASE_URL:-}" || -z "${AZURE_V3_SAS:-}" ]]; then
    echo "Set AZURE_V3_BASE_URL and AZURE_V3_SAS first." >&2
    exit 1
fi

YOLO_EXPORT="${1:?Usage: upload_v3_dataset_to_azure.sh YOLO_EXPORT COCO_EXPORT}"
COCO_EXPORT="${2:?Usage: upload_v3_dataset_to_azure.sh YOLO_EXPORT COCO_EXPORT}"
BASE="${AZURE_V3_BASE_URL%/}"

for split in train valid test; do
    case "${split}" in
        train) expected=1511 ;;
        valid) expected=248 ;;
        test) expected=263 ;;
    esac
    image_count="$(find "${YOLO_EXPORT}/${split}/images" -maxdepth 1 -type f -iname '*.jpg' | wc -l)"
    label_count="$(find "${YOLO_EXPORT}/${split}/labels" -maxdepth 1 -type f -name '*.txt' | wc -l)"
    coco_image_count="$(find "${COCO_EXPORT}/${split}" -maxdepth 1 -type f -iname '*.jpg' | wc -l)"
    test "${image_count}" -eq "${expected}"
    test "${label_count}" -eq "${expected}"
    test "${coco_image_count}" -eq "${expected}"
    test -f "${COCO_EXPORT}/${split}/_annotations.coco.json"

    echo "Uploading v3 ${split}: ${image_count} images and ${label_count} YOLO labels"
    azcopy copy \
        "${YOLO_EXPORT}/${split}/images/*" \
        "${BASE}/images/${split}${AZURE_V3_SAS}" \
        --recursive=true --put-md5=true
    azcopy copy \
        "${YOLO_EXPORT}/${split}/labels/*" \
        "${BASE}/labels/${split}${AZURE_V3_SAS}" \
        --recursive=true --put-md5=true
    azcopy copy \
        "${COCO_EXPORT}/${split}/_annotations.coco.json" \
        "${BASE}/annotations/coco/${split}.json${AZURE_V3_SAS}" \
        --put-md5=true
done

echo "V3 dataset uploaded under ${BASE}"
