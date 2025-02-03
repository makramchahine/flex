#!/bin/bash

export LOG_DIR="../local"
export HYDRA_FULL_ERROR=1

## flight data
export DATA_ROOT_DIR=".." # e.g., used for visualization
export DATA_DIR="${DATA_ROOT_DIR}/BLIP2_DATASET"
export PYOPENGL_PLATFORM=egl
export EGL_DEVICE_ID=${CUDA_VISIBLE_DEVICES:-0} # otherwise pyrender will leak to GPU0

## CLIP
export PRETRAINED_CLIP_DIR="../local/pretrained_ckpt"

## SAM
export PRETRAINED_SAM_DIR="../local/pretrained_ckpt"

## BLIP
export PRETRAINED_BLIP_DIR="${TORCH_HOME}/hub/checkpoints"
export HF_HOME="../local/pretrained_ckpt"

export PYTHONPATH=$PYTHONPATH:/home/alex/flex/gym-pybullet-drones
export PATH=$PATH:/home/alex/flex/learning/src
export PYTHONPATH=$PYTHONPATH:/home/alex/flex/learning