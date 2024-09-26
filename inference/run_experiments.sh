#!/bin/bash

n=50

# loop over tasks in "2rb", "2colors" , "2rshape", "2mshape", "3open_dict"
# and over environments in "samurai", "arena"


for task in "2rb" "2colors" "2rshape" "2mshape" "3open_dict"
do
    for env in "samurai" "arena"
    do
        echo "Running task $task in environment $env"
        for i in $(seq 1 $n)
        do
            echo -n "Running iteration $i/$n..." # Displaying current iteration
            python wrapper.py --task $task --env $env > /dev/null 2>&1
            echo " Done"
        done | pv -l -s $n > /dev/null
    done
done

