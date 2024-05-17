#!/bin/bash

CURRENT_DIR=$PWD/`dirname -- ${BASH_SOURCE}`
source $CURRENT_DIR/setup_macro.sh

args=(
    task_name=test_datamodule
    experiment=flight_blip_simple_vit
    data.mode=GPL
    ++data.buffer_size=1
)

python tests/test_datamodule.py "${args[@]}"

stty sane
