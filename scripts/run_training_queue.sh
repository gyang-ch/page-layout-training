#!/usr/bin/env bash
# Run the six nested learning-curve subsets sequentially, alternating YOLO/DINO.
# Successful runs receive a .training_complete marker. Re-running this script
# skips completed runs and resumes interrupted runs from their last checkpoint.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

YOLO_ENV="${YOLO_ENV:-page-yolo}"
DINO_ENV="${DINO_ENV:-page-dino}"
MMDET_ROOT="${MMDET_ROOT:-/workspace/mmdetection}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/workspace/outputs}"
YOLO_MODEL="${YOLO_MODEL:-yolo11m.pt}"
SEED="${SEED:-20260822}"
YOLO_EPOCHS="${YOLO_EPOCHS:-100}"
YOLO_BATCH="${YOLO_BATCH:-8}"
YOLO_IMGSZ="${YOLO_IMGSZ:-640}"
YOLO_WORKERS="${YOLO_WORKERS:-4}"
DINO_AMP="${DINO_AMP:-1}"

SIZES=(200 400 600 800 1000 1152)
QUEUE_LOG="${OUTPUT_ROOT}/training_queue.log"

mkdir -p "${OUTPUT_ROOT}/yolo" "${OUTPUT_ROOT}/dino"
exec > >(tee -a "${QUEUE_LOG}") 2>&1

on_error() {
    local status=$?
    echo "[$(date --iso-8601=seconds)] Queue stopped after an error (exit ${status})."
    echo "Correct the problem and run this script again; completed runs will be skipped."
    exit "${status}"
}
trap on_error ERR

echo "[$(date --iso-8601=seconds)] Starting page-layout training queue"
echo "Seed=${SEED}; YOLO epochs=${YOLO_EPOCHS}; DINO schedule comes from generated configs"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
df -h "${OUTPUT_ROOT}"

for size in "${SIZES[@]}"; do
    test -f "${REPO_ROOT}/.runtime/yolo/train_${size}.yaml"
    test -f "${REPO_ROOT}/.runtime/dino/train_${size}.py"
done
test -f "${MMDET_ROOT}/tools/train.py"

run_yolo() {
    local size="$1"
    local name="train_${size}_seed_${SEED}"
    local work_dir="${OUTPUT_ROOT}/yolo/${name}"
    local marker="${work_dir}/.training_complete"
    local last_checkpoint="${work_dir}/weights/last.pt"

    if [[ -f "${marker}" ]]; then
        echo "[$(date --iso-8601=seconds)] SKIP completed YOLO ${size}"
        return
    fi

    mkdir -p "${work_dir}"
    echo "[$(date --iso-8601=seconds)] START YOLO ${size}"
    if [[ -f "${last_checkpoint}" ]]; then
        echo "Resuming YOLO from ${last_checkpoint}"
        conda run --no-capture-output -n "${YOLO_ENV}" \
            yolo detect train \
            model="${last_checkpoint}" \
            resume=True \
            device=0
    else
        conda run --no-capture-output -n "${YOLO_ENV}" \
            yolo detect train \
            model="${YOLO_MODEL}" \
            data="${REPO_ROOT}/.runtime/yolo/train_${size}.yaml" \
            epochs="${YOLO_EPOCHS}" \
            patience=0 \
            imgsz="${YOLO_IMGSZ}" \
            batch="${YOLO_BATCH}" \
            device=0 \
            workers="${YOLO_WORKERS}" \
            project="${OUTPUT_ROOT}/yolo" \
            name="${name}" \
            exist_ok=True \
            seed="${SEED}" \
            deterministic=True \
            amp=True \
            cache=False \
            plots=True
    fi
    touch "${marker}"
    echo "[$(date --iso-8601=seconds)] DONE YOLO ${size}"
}

run_dino() {
    local size="$1"
    local name="train_${size}_seed_${SEED}"
    local work_dir="${OUTPUT_ROOT}/dino/${name}"
    local marker="${work_dir}/.training_complete"
    local -a command=(
        env MPLBACKEND=Agg
        conda run --no-capture-output -n "${DINO_ENV}"
        python "${MMDET_ROOT}/tools/train.py"
        "${REPO_ROOT}/.runtime/dino/train_${size}.py"
        --work-dir "${work_dir}"
    )

    if [[ -f "${marker}" ]]; then
        echo "[$(date --iso-8601=seconds)] SKIP completed DINO ${size}"
        return
    fi

    mkdir -p "${work_dir}"
    echo "[$(date --iso-8601=seconds)] START DINO ${size}"
    if [[ "${DINO_AMP}" == "1" ]]; then
        command+=(--amp)
    fi
    if [[ -f "${work_dir}/last_checkpoint" ]]; then
        echo "Resuming DINO from the latest checkpoint in ${work_dir}"
        command+=(--resume)
    fi
    "${command[@]}"
    touch "${marker}"
    echo "[$(date --iso-8601=seconds)] DONE DINO ${size}"
}

for size in "${SIZES[@]}"; do
    run_yolo "${size}"
    run_dino "${size}"
done

echo "[$(date --iso-8601=seconds)] All 12 training runs completed successfully."
df -h "${OUTPUT_ROOT}"
