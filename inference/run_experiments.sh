#!/bin/bash

n=10

for i in $(seq 1 $n)
do
    echo -n "Running iteration $i/$n..." # Displaying current iteration
    python wrapper.py > /dev/null 2>&1
    echo " Done"
done | pv -l -s $n > /dev/null
