#!/usr/bin/env python3
import math
import os
import subprocess
from mpi4py import MPI

rank = MPI.COMM_WORLD.Get_rank()
SLURMD_NODENAME = os.environ['SLURMD_NODENAME']
tasks = int(os.environ['SLURM_NNODES'])
chunks = 30

# FIXME: this job distribution is not great
chunks_per_task = math.ceil(chunks / tasks)
begin = rank * chunks_per_task
end = min(begin + chunks_per_task, chunks)
cpus = os.environ['SLURM_CPUS_PER_TASK']
print(f'rank={rank}, name={SLURMD_NODENAME}, cpus={cpus} begin={begin}, end={end}')

os.makedirs('logs', exist_ok=True)
out_file = open(f'logs/{rank}_out.txt', 'w')
subprocess.run(['./prepare_pile_data.py', 'range', str(begin), str(end)], stdout=out_file, stderr=subprocess.STDOUT)
