#!/bin/bash

TASK_NAME=${1:-"train_flight"}
CURRENT_DIR=$PWD/`dirname -- ${BASH_SOURCE}`
source $CURRENT_DIR/setup_macro.sh

# Define the number of experiments or a list of configurations
experiment_configs=("flight_blip_simple_vit"")

# Loop over each configuration and run the experiment
for experiment in "${experiment_configs[@]}"; do
    args=(
        task_name=$TASK_NAME
        experiment=$experiment
    )

    # Train the experiment
    python src/train.py "${args[@]}"

done

stty sane
