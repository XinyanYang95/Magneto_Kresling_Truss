#!/bin/bash
#SBATCH -A p31175             ## account (unchanged)
#SBATCH -p short         ## "-p" instead of "-q"
#SBATCH -N 1                 ## number of nodes
#SBATCH -n 28               ## number of cores
#SBATCH -t 3:55:00          ## walltime
#SBATCH	--job-name="minima"    ## name of job

##### These are shell commands. Note that all MSUB commands come first.
module load mpi
module load python/anaconda3
source activate /home/xys3549/anaconda/envs
mpirun -np 28 /home/xys3549/anaconda/envs/bin/python3.6 minima_mpi.py > minima_mpi.out
