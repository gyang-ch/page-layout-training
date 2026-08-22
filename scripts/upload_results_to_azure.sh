#!/usr/bin/env bash
set -euo pipefail

if ! command -v azcopy >/dev/null 2>&1; then
  echo "azcopy is required." >&2
  exit 1
fi
if [[ -z "${AZURE_RESULTS_BASE_URL:-}" || -z "${AZURE_RESULTS_SAS:-}" ]]; then
  echo "Set AZURE_RESULTS_BASE_URL and AZURE_RESULTS_SAS first." >&2
  exit 1
fi

results_dir="${1:?Usage: upload_results_to_azure.sh RESULTS_DIR RUN_NAME}"
run_name="${2:?Usage: upload_results_to_azure.sh RESULTS_DIR RUN_NAME}"
if [[ ! -d "$results_dir" ]]; then
  echo "Results directory not found: $results_dir" >&2
  exit 1
fi

destination="${AZURE_RESULTS_BASE_URL%/}/$run_name${AZURE_RESULTS_SAS}"
azcopy copy "$results_dir/*" "$destination" --recursive=true --put-md5=true
echo "Results uploaded under: ${AZURE_RESULTS_BASE_URL%/}/$run_name"

