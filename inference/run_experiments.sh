#!/bin/bash

n=5
# Define the tasks and environments you want to use
# tasks=("2rb" "2colors" "2rshape" "2mshape" "3open_dict")
# envs=("samurai" "arena")

tasks=("2rshape")
envs=("samurai")
command=("above" "below" "right" "left" "towards")

# Loop over each task and env_name
for i in "${!tasks[@]}"; do
    for j in "${!envs[@]}"; do 
        for k in "${!command[@]}"; do
            task="${tasks[$i]}"
            env="${envs[$j]}"
            cmd="${command[$k]}"
            
            echo "Running with Task: $task and Env: $env and Command: $cmd"

            # Use sed to change the task in config.py
            sed -i "s/task = Task(\"[^\"]*\")/task = Task(\"$task\")/" /home/alex/flex/inference/config.py
            # Use sed to change the env_name in /home/alex/flex/inference/config.py
            sed -i "s/env_name = \"[^\"]*\"/env_name = \"$env\"/" /home/alex/flex/inference/config.py
            # Use sed to change the command in /home/alex/flex/inference/config.py
            sed -i "s/command = \"[^\"]*\"/command = \"$cmd\"/" /home/alex/flex/inference/config.py

            for i in $(seq 1 $n)
            do
                echo -n "Running iteration $i/$n..." # Displaying current iteration
                python wrapper.py
                echo " Done"
            done | pv -l -s $n > /dev/null
        done
    done
done
