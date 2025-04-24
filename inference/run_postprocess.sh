#!/bin/bash

# for task in "2rb" "2colors" "2rshape" "2mshape" "3open_dict"
for task in "2rb" "2colors" "2rshape"
do
    for env in "samurai" "arena"
    do
        echo "Running task $task in environment $env"
        python postprocess.py --task $task --env $env
        echo " Done"
    done
done
