#!/bin/bash

CURRENT_DIR=$PWD/`dirname -- ${BASH_SOURCE}`
source $CURRENT_DIR/setup_macro.sh

args=(
    task_name=test_module
    experiment=flight_blip_simple_vit
    ++data.buffer_size=1
)

python tests/test_module.py "${args[@]}"

stty sane
