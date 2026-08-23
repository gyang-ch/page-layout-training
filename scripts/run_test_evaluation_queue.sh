#!/usr/bin/env bash
# Evaluate all best validation-selected checkpoints on the untouched test set.
# This script is restart-safe and never chooses a checkpoint using test scores.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

YOLO_ENV="${YOLO_ENV:-page-yolo}"
DINO_ENV="${DINO_ENV:-page-dino}"
MMDET_ROOT="${MMDET_ROOT:-/workspace/mmdetection}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/workspace/outputs}"
SEED="${SEED:-20260822}"
YOLO_BATCH="${YOLO_BATCH:-8}"
YOLO_IMGSZ="${YOLO_IMGSZ:-640}"
YOLO_WORKERS="${YOLO_WORKERS:-4}"

SIZES=(200 400 600 800 1000 1152)
TEST_ROOT="${OUTPUT_ROOT}/test"
QUEUE_LOG="${TEST_ROOT}/test_evaluation_queue.log"

mkdir -p "${TEST_ROOT}/yolo" "${TEST_ROOT}/dino"
exec > >(tee -a "${QUEUE_LOG}") 2>&1

on_error() {
    local status=$?
    echo "[$(date --iso-8601=seconds)] Test queue stopped (exit ${status})."
    echo "Fix the problem and rerun; completed evaluations will be skipped."
    exit "${status}"
}
trap on_error ERR

echo "[$(date --iso-8601=seconds)] Starting fixed-test evaluation queue"
echo "The test split contains 263 images and is used only for final evaluation."
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

for size in "${SIZES[@]}"; do
    test -f "${REPO_ROOT}/.runtime/yolo/train_${size}.yaml"
    test -f "${REPO_ROOT}/.runtime/dino/train_${size}.py"
    test -f "${OUTPUT_ROOT}/yolo/train_${size}_seed_${SEED}/weights/best.pt"
done
test -f "${MMDET_ROOT}/tools/test.py"

run_yolo_test() {
    local size="$1"
    local train_dir="${OUTPUT_ROOT}/yolo/train_${size}_seed_${SEED}"
    local test_dir="${TEST_ROOT}/yolo/train_${size}_seed_${SEED}"
    local marker="${test_dir}/.evaluation_complete"

    if [[ -f "${marker}" ]]; then
        echo "[$(date --iso-8601=seconds)] SKIP completed YOLO test ${size}"
        return
    fi

    echo "[$(date --iso-8601=seconds)] START YOLO test ${size}"
    conda run --no-capture-output -n "${YOLO_ENV}" \
        python "${SCRIPT_DIR}/evaluate_yolo_test.py" \
        --model "${train_dir}/weights/best.pt" \
        --data "${REPO_ROOT}/.runtime/yolo/train_${size}.yaml" \
        --output-dir "${test_dir}" \
        --imgsz "${YOLO_IMGSZ}" \
        --batch "${YOLO_BATCH}" \
        --workers "${YOLO_WORKERS}"
    touch "${marker}"
    echo "[$(date --iso-8601=seconds)] DONE YOLO test ${size}"
}

run_dino_test() {
    local size="$1"
    local train_dir="${OUTPUT_ROOT}/dino/train_${size}_seed_${SEED}"
    local test_dir="${TEST_ROOT}/dino/train_${size}_seed_${SEED}"
    local marker="${test_dir}/.evaluation_complete"
    local -a checkpoints=()

    if [[ -f "${marker}" ]]; then
        echo "[$(date --iso-8601=seconds)] SKIP completed DINO test ${size}"
        return
    fi

    mapfile -t checkpoints < <(
        find "${train_dir}" -maxdepth 1 -type f \
            -name 'best_coco_bbox_mAP_epoch_*.pth' | sort
    )
    if [[ "${#checkpoints[@]}" -ne 1 ]]; then
        echo "Expected exactly one best DINO checkpoint in ${train_dir}; found ${#checkpoints[@]}" >&2
        return 1
    fi

    mkdir -p "${test_dir}"
    echo "[$(date --iso-8601=seconds)] START DINO test ${size}"
    env MPLBACKEND=Agg conda run --no-capture-output -n "${DINO_ENV}" \
        python "${MMDET_ROOT}/tools/test.py" \
        "${REPO_ROOT}/.runtime/dino/train_${size}.py" \
        "${checkpoints[0]}" \
        --work-dir "${test_dir}" \
        --out "${test_dir}/predictions.pkl" \
        --cfg-options \
        test_evaluator.outfile_prefix="${test_dir}/coco_predictions"
    touch "${marker}"
    echo "[$(date --iso-8601=seconds)] DONE DINO test ${size}"
}

for size in "${SIZES[@]}"; do
    run_yolo_test "${size}"
    run_dino_test "${size}"
done

echo "[$(date --iso-8601=seconds)] All 12 fixed-test evaluations completed."
