#!/usr/bin/env bash
# Run the new v3 YOLO and DINO checkpoints on Bodleian books that were not
# used in v3 training. The selected book directories are staged as symlinks so
# relative book/image paths remain identical to the earlier inference package.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

SOURCE_ROOT="${SOURCE_ROOT:-/workspace/bodleian}"
INPUT_ROOT="${INPUT_ROOT:-/workspace/bodleian_unseen_v3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/workspace/outputs_v3/bodleian_unseen_v3_new_models_seed_20260822}"
MANIFEST="${MANIFEST:-${REPO_ROOT}/config/bodleian_unseen_v3_subfolders.txt}"
YOLO_ENV="${YOLO_ENV:-page-yolo}"
DINO_ENV="${DINO_ENV:-page-dino}"
CONFIDENCE="${CONFIDENCE:-0.25}"
YOLO_BATCH="${YOLO_BATCH:-8}"
YOLO_IMGSZ="${YOLO_IMGSZ:-640}"

CONDA_EXE="${CONDA_EXE:-/workspace/miniconda/bin/conda}"
if [[ ! -x "${CONDA_EXE}" ]]; then
    CONDA_EXE="$(command -v conda || true)"
fi
if [[ -z "${CONDA_EXE}" || ! -x "${CONDA_EXE}" ]]; then
    echo "Conda executable not found." >&2
    exit 1
fi

YOLO_CHECKPOINT="/workspace/outputs_v3/yolo/train_1511_from_1152_seed_20260822/weights/best.pt"
DINO_CHECKPOINT="/workspace/outputs_v3/dino/train_1511_from_1152_seed_20260822/best_coco_bbox_mAP_epoch_4.pth"
DINO_CONFIG="${REPO_ROOT}/.runtime_v3/dino/train_1511_from_1152.py"

for required in "${MANIFEST}" "${YOLO_CHECKPOINT}" "${DINO_CHECKPOINT}" "${DINO_CONFIG}"; do
    if [[ ! -e "${required}" ]]; then
        echo "Missing required file: ${required}" >&2
        exit 1
    fi
done

mkdir -p "${INPUT_ROOT}" "${OUTPUT_ROOT}"
folder_count=0
while IFS= read -r folder || [[ -n "${folder}" ]]; do
    [[ -z "${folder}" || "${folder}" == \#* ]] && continue
    source_dir="${SOURCE_ROOT}/${folder}"
    if [[ ! -d "${source_dir}" ]]; then
        echo "Manifest subfolder missing from Bodleian corpus: ${folder}" >&2
        exit 1
    fi
    ln -sfn "${source_dir}" "${INPUT_ROOT}/${folder}"
    folder_count=$((folder_count + 1))
done < "${MANIFEST}"

if [[ "${folder_count}" -ne 45 ]]; then
    echo "Expected 45 manifest subfolders; found ${folder_count}." >&2
    exit 1
fi

image_count="$(find -L "${INPUT_ROOT}" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.tif' -o -iname '*.tiff' -o -iname '*.webp' \) | wc -l | tr -d ' ')"
if [[ "${image_count}" -eq 0 ]]; then
    echo "No images found under ${INPUT_ROOT}." >&2
    exit 1
fi

LOG_FILE="${OUTPUT_ROOT}/inference_queue.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

on_error() {
    local status=$?
    echo "[$(date --iso-8601=seconds)] Queue stopped with exit ${status}. Rerun after correcting the problem."
    exit "${status}"
}
trap on_error ERR

echo "[$(date --iso-8601=seconds)] Starting v3 inference on unseen Bodleian subfolders"
echo "Subfolders: ${folder_count}; images: ${image_count}; confidence: ${CONFIDENCE}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

YOLO_OUTPUT="${OUTPUT_ROOT}/yolo/new_1511_on_unseen_bodleian"
if [[ ! -f "${YOLO_OUTPUT}/.inference_complete" ]]; then
    mkdir -p "${YOLO_OUTPUT}"
    "${CONDA_EXE}" run --no-capture-output -n "${YOLO_ENV}" \
        python "${SCRIPT_DIR}/predict_external_yolo.py" \
        --model "${YOLO_CHECKPOINT}" \
        --input-root "${INPUT_ROOT}" \
        --output-dir "${YOLO_OUTPUT}" \
        --confidence "${CONFIDENCE}" \
        --imgsz "${YOLO_IMGSZ}" \
        --batch-size "${YOLO_BATCH}"
    touch "${YOLO_OUTPUT}/.inference_complete"
else
    echo "SKIP completed new YOLO-1511 inference"
fi

DINO_OUTPUT="${OUTPUT_ROOT}/dino/new_1511_on_unseen_bodleian"
if [[ ! -f "${DINO_OUTPUT}/.inference_complete" ]]; then
    mkdir -p "${DINO_OUTPUT}"
    env MPLBACKEND=Agg "${CONDA_EXE}" run --no-capture-output -n "${DINO_ENV}" \
        python "${SCRIPT_DIR}/predict_external_dino.py" \
        --config "${DINO_CONFIG}" \
        --checkpoint "${DINO_CHECKPOINT}" \
        --input-root "${INPUT_ROOT}" \
        --output-dir "${DINO_OUTPUT}" \
        --confidence "${CONFIDENCE}"
    touch "${DINO_OUTPUT}/.inference_complete"
else
    echo "SKIP completed new DINO-1511 inference"
fi

for result_dir in "${YOLO_OUTPUT}" "${DINO_OUTPUT}"; do
    annotated_count="$(find "${result_dir}/annotated" -type f | wc -l | tr -d ' ')"
    prediction_count="$(wc -l < "${result_dir}/predictions.jsonl" | tr -d ' ')"
    if [[ "${annotated_count}" -ne "${image_count}" || "${prediction_count}" -ne "${image_count}" ]]; then
        echo "Output count mismatch in ${result_dir}: annotated=${annotated_count}, predictions=${prediction_count}, expected=${image_count}" >&2
        exit 1
    fi
done

cp "${MANIFEST}" "${OUTPUT_ROOT}/bodleian_unseen_v3_subfolders.txt"
echo "[$(date --iso-8601=seconds)] Both new-model inference runs completed and validated."
