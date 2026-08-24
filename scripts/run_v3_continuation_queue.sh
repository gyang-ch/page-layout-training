#!/usr/bin/env bash
# Fine-tune the old 1,152-image best YOLO and DINO weights on the complete
# 1,511-image v3 dataset. New output directories preserve all old experiments.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

YOLO_ENV="${YOLO_ENV:-page-yolo}"
DINO_ENV="${DINO_ENV:-page-dino}"
MMDET_ROOT="${MMDET_ROOT:-/workspace/mmdetection}"
OLD_OUTPUT_ROOT="${OLD_OUTPUT_ROOT:-/workspace/outputs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/workspace/outputs_v3}"
SEED="${SEED:-20260822}"
YOLO_EPOCHS="${YOLO_EPOCHS:-50}"
YOLO_BATCH="${YOLO_BATCH:-8}"
YOLO_IMGSZ="${YOLO_IMGSZ:-640}"
YOLO_WORKERS="${YOLO_WORKERS:-4}"
YOLO_LR0="${YOLO_LR0:-0.001}"
DINO_AMP="${DINO_AMP:-1}"

YOLO_CONFIG="${REPO_ROOT}/.runtime_v3/yolo/train_1511_from_1152.yaml"
DINO_CONFIG="${REPO_ROOT}/.runtime_v3/dino/train_1511_from_1152.py"
YOLO_INIT="${OLD_OUTPUT_ROOT}/yolo/train_1152_seed_${SEED}/weights/best.pt"
DINO_OLD_DIR="${OLD_OUTPUT_ROOT}/dino/train_1152_seed_${SEED}"
NAME="train_1511_from_1152_seed_${SEED}"
LOG_FILE="${OUTPUT_ROOT}/v3_continuation_training_queue.log"

if command -v conda >/dev/null 2>&1; then
    CONDA_BIN="$(command -v conda)"
elif [[ -x /workspace/miniconda/bin/conda ]]; then
    CONDA_BIN=/workspace/miniconda/bin/conda
else
    echo "Conda was not found in PATH or /workspace/miniconda/bin/conda." >&2
    exit 1
fi

mkdir -p "${OUTPUT_ROOT}/yolo" "${OUTPUT_ROOT}/dino"
exec > >(tee -a "${LOG_FILE}") 2>&1

on_error() {
    local status=$?
    echo "[$(date --iso-8601=seconds)] V3 queue stopped (exit ${status})."
    echo "Correct the problem and rerun; interrupted new runs will resume."
    exit "${status}"
}
trap on_error ERR

test -f "${YOLO_CONFIG}"
test -f "${DINO_CONFIG}"
test -f "${YOLO_INIT}"
test -f "${MMDET_ROOT}/tools/train.py"
mkdir -p "${OUTPUT_ROOT}/provenance"
cp "${YOLO_CONFIG}" "${OUTPUT_ROOT}/provenance/"
cp "${DINO_CONFIG}" "${OUTPUT_ROOT}/provenance/"
cp "${REPO_ROOT}/.runtime_v3/v3_runtime_summary.json" "${OUTPUT_ROOT}/provenance/"
mapfile -t DINO_INIT_MATCHES < <(
    find "${DINO_OLD_DIR}" -maxdepth 1 -type f \
        -name 'best_coco_bbox_mAP_epoch_*.pth' | sort
)
if [[ "${#DINO_INIT_MATCHES[@]}" -ne 1 ]]; then
    echo "Expected exactly one old best DINO checkpoint; found ${#DINO_INIT_MATCHES[@]}." >&2
    exit 1
fi

echo "[$(date --iso-8601=seconds)] Starting v3 continuation training"
echo "YOLO: 50 epochs by default, lr0=${YOLO_LR0}; DINO schedule is in ${DINO_CONFIG}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
df -h "${OUTPUT_ROOT}"

YOLO_WORK_DIR="${OUTPUT_ROOT}/yolo/${NAME}"
if [[ -f "${YOLO_WORK_DIR}/.training_complete" ]]; then
    echo "SKIP completed v3 YOLO"
else
    mkdir -p "${YOLO_WORK_DIR}"
    if [[ -f "${YOLO_WORK_DIR}/weights/last.pt" ]]; then
        echo "Resuming interrupted v3 YOLO run"
        "${CONDA_BIN}" run --no-capture-output -n "${YOLO_ENV}" \
            yolo detect train model="${YOLO_WORK_DIR}/weights/last.pt" resume=True device=0
    else
        "${CONDA_BIN}" run --no-capture-output -n "${YOLO_ENV}" \
            yolo detect train \
            model="${YOLO_INIT}" \
            data="${YOLO_CONFIG}" \
            epochs="${YOLO_EPOCHS}" patience=0 \
            imgsz="${YOLO_IMGSZ}" batch="${YOLO_BATCH}" \
            workers="${YOLO_WORKERS}" device=0 \
            project="${OUTPUT_ROOT}/yolo" name="${NAME}" exist_ok=True \
            seed="${SEED}" deterministic=True amp=True cache=False plots=True \
            lr0="${YOLO_LR0}" cos_lr=True close_mosaic=10
    fi
    touch "${YOLO_WORK_DIR}/.training_complete"
fi

DINO_WORK_DIR="${OUTPUT_ROOT}/dino/${NAME}"
if [[ -f "${DINO_WORK_DIR}/.training_complete" ]]; then
    echo "SKIP completed v3 DINO"
else
    mkdir -p "${DINO_WORK_DIR}"
    DINO_COMMAND=(
        env MPLBACKEND=Agg
        "${CONDA_BIN}" run --no-capture-output -n "${DINO_ENV}"
        python "${MMDET_ROOT}/tools/train.py"
        "${DINO_CONFIG}"
        --work-dir "${DINO_WORK_DIR}"
    )
    if [[ "${DINO_AMP}" == "1" ]]; then
        DINO_COMMAND+=(--amp)
    fi
    if [[ -f "${DINO_WORK_DIR}/last_checkpoint" ]]; then
        echo "Resuming interrupted v3 DINO run"
        DINO_COMMAND+=(--resume)
    fi
    "${DINO_COMMAND[@]}"
    touch "${DINO_WORK_DIR}/.training_complete"
fi

echo "[$(date --iso-8601=seconds)] Both v3 continuation runs completed."
df -h "${OUTPUT_ROOT}"
