#!/usr/bin/env bash
set -euo pipefail

if ! command -v azcopy >/dev/null 2>&1; then
  echo "azcopy is required: https://learn.microsoft.com/azure/storage/common/storage-use-azcopy-v10" >&2
  exit 1
fi

if [[ -z "${AZURE_DATASET_BASE_URL:-}" || -z "${AZURE_DATASET_SAS:-}" ]]; then
  echo "Set AZURE_DATASET_BASE_URL and AZURE_DATASET_SAS first; see .env.example." >&2
  exit 1
fi

dataset_root="${1:-/workspace/datasets/page-layout-v2}"
base="${AZURE_DATASET_BASE_URL%/}"

for split in train valid test; do
  mkdir -p "$dataset_root/images/$split" "$dataset_root/labels/$split"
  echo "Downloading $split images..."
  azcopy copy "$base/images/$split/*${AZURE_DATASET_SAS}" "$dataset_root/images/$split" --recursive=true --check-md5=FailIfDifferent
  echo "Downloading $split labels..."
  azcopy copy "$base/labels/$split/*${AZURE_DATASET_SAS}" "$dataset_root/labels/$split" --recursive=true --check-md5=FailIfDifferent
done

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "$script_dir/verify_dataset.py" --dataset-root "$dataset_root"
echo "Dataset ready at: $dataset_root"

