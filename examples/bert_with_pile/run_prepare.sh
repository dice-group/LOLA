#!/bin/bash
#echo "Current Node: ${SLURMD_NODENAME}"
#echo "Chunk offset: ${CHUNK_OFFSET}"
TASK_RANK=`python get_task_rank.py`
#echo "Task rank: ${TASK_RANK}"
# Compute start and end range
STRT_CHUNK=$((TASK_RANK * CHUNK_OFFSET))
END_CHUNK=$((STRT_CHUNK + CHUNK_OFFSET)) 
if [ ${TASK_RANK} -eq $((TOTAL_TASKS - 1)) ]; then
    #echo "${TASK_RANK} is the last task"
    END_CHUNK=30
fi

echo "Current Node ${SLURMD_NODENAME} with rank ${TASK_RANK} has range ${STRT_CHUNK} ${END_CHUNK}"
python prepare_pile_data.py range $STRT_CHUNK $END_CHUNK  2>&1 | tee logs/${TASK_RANK}_out.txt
