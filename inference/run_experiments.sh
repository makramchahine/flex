# script that just runs the python wrapper script n times.
n=10

for i in $(seq 1 $n)
do
    python wrapper.py
done