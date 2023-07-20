#!/bin/bash

if [[ "$1" != "" && "$1" -le 8 ]]; then
    NGPU=$1
else
    echo "Defaulting to 1 GPU."
    NGPU=1
fi
TIME=$((270-30*$NGPU))
HOURS=$(($TIME/60))
MINUTES=$(($TIME%60))
CORES_PER_GPU=16
CORES=$(($CORES_PER_GPU * $NGPU))
printf "Requesting %d GPUs (+ %d CPU-cores) for %d minutes (%02d:%02d:00)\n\n" $NGPU $CORES $TIME $HOURS $MINUTES
srun -N 1 -n 1 -c $CORES --gres=gpu:a100:$NGPU --qos=devel -p dgx -t $TIME --pty bash
