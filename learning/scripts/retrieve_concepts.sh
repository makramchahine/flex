#!/bin/bash

### Related Checkpoints
# train_vista_gpl/2023-07-26/12-55-21: [OA] GPL-OA; BLIP FE; Simple ViT Policy; no ROI; masked patch feature
###

EVAL_CKPT_DIR=${1:-"../local/train_vista_gpl/2023-07-26/12-55-21"}
SAVE_FEATURES_PATH="./outputs/2023-08-21/19-17-59/feats.pkl" # change this
ANALYSIS_OUT_DIR="./outputs/retrieved_concepts"
TASK_NAME="null"

CURRENT_DIR=$PWD/`dirname -- ${BASH_SOURCE}`
source $CURRENT_DIR/setup_macro.sh

source $CURRENT_DIR/base_eval_args.sh
args=("${BASE_VISTA_EVAL_ARGS_OA[@]}")

args+=(
    task_name=$TASK_NAME # not used
    +ckpt_path="$EVAL_CKPT_DIR/checkpoints/last.ckpt" # with respect to this script
    hydra.job.chdir=False
    +saved_features_path=$SAVE_FEATURES_PATH
    +analysis_out_dir=$ANALYSIS_OUT_DIR
    ## disable logging
    hydra.run.dir=.
    hydra.output_subdir=null
    hydra/job_logging=disabled
    hydra/hydra_logging=disabled
    ##
    --config-path "../$EVAL_CKPT_DIR/.hydra" # with respect to src/eval.py
    --config-name "config.yaml"
)

python src/retrieve_concepts.py "${args[@]}"

stty sane
