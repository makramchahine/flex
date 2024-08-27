#!/bin/bash

TASK_NAME=${1:-"train_flight"}

CURRENT_DIR=$PWD/`dirname -- ${BASH_SOURCE}`
source $CURRENT_DIR/setup_macro.sh

args=(
    task_name=$TASK_NAME
    experiment=flight_blip_simple_vit
)

python src/train.py "${args[@]}"

stty sane
