#!/bin/bash

CURRENT_DIR=$PWD/`dirname -- ${BASH_SOURCE}`
source $CURRENT_DIR/setup_macro.sh

args=(
    task_name=test_model
    experiment=flight_blip_simple_vit
    ++data.buffer_size=1
)

python tests/test_model.py "${args[@]}"

stty sane
