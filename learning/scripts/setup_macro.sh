#!/bin/bash

export LOG_DIR="../local"
export HYDRA_FULL_ERROR=1

## vista
# export DATA_ROOT_DIR="/johnson/data/vista" # in local filesystem --> faster
export DATA_ROOT_DIR=".." # e.g., used for visualization
export DATA_DIR="${DATA_ROOT_DIR}/BLIP2_DATASET"
export PYOPENGL_PLATFORM=egl
export EGL_DEVICE_ID=${CUDA_VISIBLE_DEVICES:-0} # otherwise pyrender will leak to GPU0

## DINO
export DINO_SAM_ROOT="$PWD/../dino_sam_feature_extraction/"
export PYTHONPATH="${PYTHONPATH}:${DINO_SAM_ROOT}"
export PYTHONPATH="${PYTHONPATH}:${DINO_SAM_ROOT}/Segment-and-Track-Anything"
export PYTHONPATH="${PYTHONPATH}:${DINO_SAM_ROOT}/Segment-and-Track-Anything/aot"
export PYTHONPATH="${PYTHONPATH}:${DINO_SAM_ROOT}/hub/facebookresearch_dino_main"
export TORCH_HOME="${DINO_SAM_ROOT}"

## CLIP
export PRETRAINED_CLIP_DIR="../local/pretrained_ckpt"

## BLIP
export PRETRAINED_BLIP_DIR="${TORCH_HOME}/hub/checkpoints"
export HF_HOME="../local/pretrained_ckpt"

## SAM
export PRETRAINED_SAM_DIR="../local/pretrained_ckpt"

## Objaverse
export OBJAVERSE_BASE_PATH="$DATA_ROOT_DIR/../objaverse"
