#!/usr/bin/env bash
set -euo pipefail

# Upload one normalized copy of each image and YOLO label from a Roboflow YOLO
# export. AZURE_DATASET_BASE_URL must not contain the SAS query string.

if ! command -v azcopy >/dev/null 2>&1; then
  echo "azcopy is required: https://learn.microsoft.com/azure/storage/common/storage-use-azcopy-v10" >&2
  exit 1
fi

if [[ -z "${AZURE_DATASET_BASE_URL:-}" || -z "${AZURE_DATASET_SAS:-}" ]]; then
  echo "Set AZURE_DATASET_BASE_URL and AZURE_DATASET_SAS first; see .env.example." >&2
  exit 1
fi

source_root="${1:-../Page_layout.v2i.yolov11}"
if [[ ! -d "$source_root/train/images" ]]; then
  echo "YOLO source export not found at: $source_root" >&2
  exit 1
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "$script_dir/verify_dataset.py" \
  --dataset-root "$source_root" \
  --layout roboflow-yolo

base="${AZURE_DATASET_BASE_URL%/}"
for split in train valid test; do
  echo "Uploading $split images..."
  azcopy copy "$source_root/$split/images/*" "$base/images/$split${AZURE_DATASET_SAS}" --recursive=true --put-md5=true
  echo "Uploading $split labels..."
  azcopy copy "$source_root/$split/labels/*" "$base/labels/$split${AZURE_DATASET_SAS}" --recursive=true --put-md5=true
done

echo "Upload complete: $base"
