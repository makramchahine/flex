#!/bin/bash

export LOG_DIR="../local"
export HYDRA_FULL_ERROR=1

## flight data
export DATA_ROOT_DIR="/home/gridsan/wyang/flex" # e.g., used for visualization
export DATA_DIR="${DATA_ROOT_DIR}/BLIP2_DATASET_1O_1C"
export PYOPENGL_PLATFORM=egl
export EGL_DEVICE_ID=${CUDA_VISIBLE_DEVICES:-0} # otherwise pyrender will leak to GPU0

## CLIP
export PRETRAINED_CLIP_DIR="../local/pretrained_ckpt"

## SAM
export PRETRAINED_SAM_DIR="../local/pretrained_ckpt"

## BLIP
export PRETRAINED_BLIP_DIR="${TORCH_HOME}/hub/checkpoints"
export HF_HOME="../local/pretrained_ckpt"

## LOG EXPORTS
export LOG_DIR="../logs"

## Setup for SuperCloud Only
HF_USER_DIR="/home/gridsan/$(whoami)/.cache/huggingface"
HF_LOCAL_DIR="/state/partition1/user/$(whoami)/cache/huggingface"
mkdir -p $HF_LOCAL_DIR
rsync -a --ignore-existing $HF_USER_DIR/ ${HF_LOCAL_DIR}
export HF_HOME=${HF_LOCAL_DIR}