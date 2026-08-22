# Page Layout learning-curve training package

This repository contains the small, version-controlled portion of the dataset:
selection manifests, COCO annotations, integrity metadata, and deployment tools.
Images and YOLO labels are stored once in Azure Blob Storage and downloaded to
RunPod. The same image files are used by YOLO11 and DINO/MMDetection.

## Dataset invariants

- Selection seed: `20260822`
- Nested train sizes: `200, 400, 600, 800, 1000, 1152`
- Fixed validation set: 248 images
- Fixed test set: 263 images
- Classes: `illustration`, `text_block`

The original filename-only manifests are under `manifests/`. The COCO subset
files and original full/fixed split annotations are under `annotations/coco/`.
`metadata/dataset_inventory.json` records the expected size and SHA-256 of every
unique image and YOLO label.

## Publish this package to GitHub

This local directory has already been initialized as a Git repository. Create an
empty private repository named `page-layout-training` in GitHub. From this
directory, run:

```bash
git add .
git commit -m "Add reproducible page-layout training package"
git branch -M main
git remote add origin https://github.com/YOUR_ACCOUNT/page-layout-training.git
git push -u origin main
```

Check `git status` before every push. Dataset images, credentials, model weights,
runtime configs, and training outputs are excluded by `.gitignore`.

## Azure layout

Upload the original YOLO export once using this normalized Blob layout:

```text
page-layout/v2-20260822/
├── images/
│   ├── train/
│   ├── valid/
│   └── test/
└── labels/
    ├── train/
    ├── valid/
    └── test/
```

Copy `.env.example` to `.env` only as a private local reference. Do not source
an untrusted `.env` file and never commit credentials. Export the variables in
your shell, install AzCopy using Microsoft's official instructions, then upload:

```bash
export AZURE_DATASET_BASE_URL='https://ACCOUNT.blob.core.windows.net/CONTAINER/page-layout/v2-20260822'
export AZURE_DATASET_SAS='?YOUR_WRITE_SAS'
./scripts/upload_dataset_to_azure.sh ../Page_layout.v2i.yolov11
```

Use a private container and a narrowly scoped, short-lived SAS. A write SAS is
needed locally for upload; RunPod should receive a separate read-only SAS.

## RunPod setup

Attach a persistent network volume if you want the dataset, model cache, and
checkpoints to survive Pod replacement. In the Pod terminal:

```bash
cd /workspace
git clone https://github.com/YOUR_ACCOUNT/page-layout-training.git
cd page-layout-training

export AZURE_DATASET_BASE_URL='https://ACCOUNT.blob.core.windows.net/CONTAINER/page-layout/v2-20260822'
export AZURE_DATASET_SAS='?YOUR_READ_ONLY_SAS'
./scripts/download_dataset_from_azure.sh /workspace/datasets/page-layout-v2
```

The download script performs a full SHA-256 verification after AzCopy finishes.

Generate YOLO configs and filename lists:

```bash
python3 scripts/prepare_runtime.py \
  --dataset-root /workspace/datasets/page-layout-v2
```

The generated configs are placed in `.runtime/yolo/`. Example:

```bash
yolo detect train \
  model=yolo11m.pt \
  data=.runtime/yolo/train_200.yaml \
  project=/workspace/outputs/yolo \
  name=train_200_seed_20260822
```

Repeat with `train_400.yaml` through `train_1152.yaml`. Every generated YOLO
configuration points validation and testing at the original fixed directories.

For DINO, first clone/install the MMDetection version you intend to use and
identify its DINO model config. Download the matching COCO-pretrained checkpoint,
then generate overlay configs that inherit from the base configuration:

```bash
python3 scripts/prepare_runtime.py \
  --dataset-root /workspace/datasets/page-layout-v2 \
  --dino-base-config /workspace/mmdetection/configs/dino/dino-4scale_r50_8xb2-12e_coco.py \
  --dino-checkpoint /workspace/models/dino-4scale_r50_12e_coco.pth \
  --seed 20260822
```

Then train from the MMDetection checkout, for example:

```bash
cd /workspace/mmdetection
python tools/train.py \
  /workspace/page-layout-training/.runtime/dino/train_200.py \
  --work-dir /workspace/outputs/dino/train_200_seed_20260822
```

The generated DINO overlays keep the requested random seed but set
`randomness.deterministic=False`. Strict deterministic mode is incompatible
with the CUDA `torch.cumsum` operation used by DINO positional encoding in the
supported PyTorch stack. Use the same seed, software environment, and GPU type
for all learning-curve runs, and record this limitation in the experiment log.

The DINO overlays use the corresponding subset JSON while all six experiments
share `images/train/`. Validation and testing use the original fixed COCO JSONs.

## Preserve results

Do not rely solely on an ephemeral Pod filesystem. Upload completed or periodic
checkpoints with a separate write SAS:

```bash
export AZURE_RESULTS_BASE_URL='https://ACCOUNT.blob.core.windows.net/CONTAINER/page-layout/results'
export AZURE_RESULTS_SAS='?YOUR_RESULTS_WRITE_SAS'
./scripts/upload_results_to_azure.sh \
  /workspace/outputs/yolo/train_200_seed_20260822 \
  yolo/train_200_seed_20260822
```

## Refreshing metadata locally

If the source export or subset generator intentionally changes, regenerate the
Git-tracked metadata from the parent workspace:

```bash
python3 scripts/export_metadata.py
```

Review the diff before committing. A fixed dataset should not produce changes.
