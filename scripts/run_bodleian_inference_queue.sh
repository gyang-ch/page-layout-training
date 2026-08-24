#!/usr/bin/env bash
# Run four validation-selected page-layout checkpoints on the Bodleian corpus.
# The queue is restart-safe and preserves book subdirectories in every output.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

INPUT_ROOT="${INPUT_ROOT:-/workspace/bodleian}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/workspace/outputs/bodleian_inference_seed_20260822}"
TRAINING_ROOT="${TRAINING_ROOT:-/workspace/outputs}"
YOLO_ENV="${YOLO_ENV:-page-yolo}"
DINO_ENV="${DINO_ENV:-page-dino}"
SEED="${SEED:-20260822}"
CONFIDENCE="${CONFIDENCE:-0.25}"
YOLO_BATCH="${YOLO_BATCH:-8}"
YOLO_IMGSZ="${YOLO_IMGSZ:-640}"
EXPECTED_IMAGES="${EXPECTED_IMAGES:-3090}"

SIZES=(600 1152)
LOG_FILE="${OUTPUT_ROOT}/inference_queue.log"
mkdir -p "${OUTPUT_ROOT}"
exec > >(tee -a "${LOG_FILE}") 2>&1

on_error() {
    local status=$?
    echo "[$(date --iso-8601=seconds)] Inference queue stopped (exit ${status})."
    echo "Fix the problem and rerun; completed models will be skipped."
    exit "${status}"
}
trap on_error ERR

image_count="$(find "${INPUT_ROOT}" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.tif' -o -iname '*.tiff' -o -iname '*.webp' \) | wc -l)"
if [[ "${image_count}" -ne "${EXPECTED_IMAGES}" ]]; then
    echo "Expected ${EXPECTED_IMAGES} images under ${INPUT_ROOT}; found ${image_count}." >&2
    exit 1
fi

echo "[$(date --iso-8601=seconds)] Starting four-model Bodleian inference queue"
echo "Input images: ${image_count}; confidence threshold: ${CONFIDENCE}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

for size in "${SIZES[@]}"; do
    test -f "${TRAINING_ROOT}/yolo/train_${size}_seed_${SEED}/weights/best.pt"
    test -f "${REPO_ROOT}/.runtime/dino/train_${size}.py"
done

run_yolo() {
    local size="$1"
    local result_dir="${OUTPUT_ROOT}/yolo/train_${size}_seed_${SEED}"
    local marker="${result_dir}/.inference_complete"
    if [[ -f "${marker}" ]]; then
        echo "SKIP completed YOLO-${size}"
        return
    fi
    mkdir -p "${result_dir}"
    conda run --no-capture-output -n "${YOLO_ENV}" \
        python "${SCRIPT_DIR}/predict_external_yolo.py" \
        --model "${TRAINING_ROOT}/yolo/train_${size}_seed_${SEED}/weights/best.pt" \
        --input-root "${INPUT_ROOT}" \
        --output-dir "${result_dir}" \
        --confidence "${CONFIDENCE}" \
        --imgsz "${YOLO_IMGSZ}" \
        --batch-size "${YOLO_BATCH}"
    touch "${marker}"
}

run_dino() {
    local size="$1"
    local train_dir="${TRAINING_ROOT}/dino/train_${size}_seed_${SEED}"
    local result_dir="${OUTPUT_ROOT}/dino/train_${size}_seed_${SEED}"
    local marker="${result_dir}/.inference_complete"
    local -a checkpoints=()
    if [[ -f "${marker}" ]]; then
        echo "SKIP completed DINO-${size}"
        return
    fi
    mapfile -t checkpoints < <(
        find "${train_dir}" -maxdepth 1 -type f \
            -name 'best_coco_bbox_mAP_epoch_*.pth' | sort
    )
    if [[ "${#checkpoints[@]}" -ne 1 ]]; then
        echo "Expected exactly one best DINO checkpoint in ${train_dir}; found ${#checkpoints[@]}." >&2
        return 1
    fi
    mkdir -p "${result_dir}"
    env MPLBACKEND=Agg conda run --no-capture-output -n "${DINO_ENV}" \
        python "${SCRIPT_DIR}/predict_external_dino.py" \
        --config "${REPO_ROOT}/.runtime/dino/train_${size}.py" \
        --checkpoint "${checkpoints[0]}" \
        --input-root "${INPUT_ROOT}" \
        --output-dir "${result_dir}" \
        --confidence "${CONFIDENCE}"
    touch "${marker}"
}

run_yolo 600
run_yolo 1152
run_dino 600
run_dino 1152

for family in yolo dino; do
    for size in "${SIZES[@]}"; do
        count="$(find "${OUTPUT_ROOT}/${family}/train_${size}_seed_${SEED}/annotated" -type f | wc -l)"
        if [[ "${count}" -ne "${EXPECTED_IMAGES}" ]]; then
            echo "Annotated-image count mismatch for ${family}-${size}: ${count}" >&2
            exit 1
        fi
    done
done

echo "[$(date --iso-8601=seconds)] All four inference runs completed and validated."
