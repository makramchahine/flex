#!/bin/bash

TASK_NAME=${1:-"train_flight"}

CURRENT_DIR=$PWD/`dirname -- ${BASH_SOURCE}`
source $CURRENT_DIR/setup_macro.sh

args=(
    task_name=$TASK_NAME
    experiment=flight_blip_svit2
)

python src/train.py "${args[@]}"

stty sane
