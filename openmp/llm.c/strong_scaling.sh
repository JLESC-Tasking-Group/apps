#!/bin/bash

# Going to execute strong scaling test with bash script. 
# After, will run a Python Script to graph.

# Cure

# Hardware:
# Intel Xeon Platinum 8180M @ 2.50GHz:
#   28 cores
#   56 threads

# Qty 13 – Intel S2600WF, (12 – 2x Intel Xeon Platinum 8180M CPU @ 2.50GHz, 1 2x Intel Xeon Platinum 8176 CPU @ 2.10GHz)
#   56 physical cores 
#   112 logical threads

# NUMA:                        
#   NUMA node(s):              2
#   NUMA node0 CPU(s):         0-27,56-83
#   NUMA node1 CPU(s):         28-55,84-111



# Notes:

# OMP_PROC_BIND specifies a binding policy which basically sets criteria by which the threads are distributed.

# OMP_PLACES: This variable can hold two kinds of values: a name specifying (hardware) places, or a list that marks places.
#   threads	a place is a single hardware thread, i. e. the hyperthreading will be ignored
#   cores	a place is a single core with its corresponding amount of hardware threads
#   sockets	a place is a single socket

RUNS=4
COUNTS="1 2 4 8 16 24"

FULL_PATH="/vast/users/tbitsky/llm.c/"
OUTPUT="$FULL_PATH/scaling_tests/strong_results_llvm22.csv"

# Clear out the previous results
: > $OUTPUT
echo "thread_num,trial_num,time" >> $OUTPUT

for count in $COUNTS;
do
    export OMP_NUM_THREADS=$count
    export OMP_PLACES="cores($count)" 
    export OMP_PROC_BIND="close"

    for run in $(seq 1 $RUNS);
    do 
        EXE_OUTPUT=$(cd $FULL_PATH && ./train_gpt2)
        echo "$count,$run,$EXE_OUTPUT" >> "$OUTPUT"
    done
done
