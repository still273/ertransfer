#!/bin/bash

# See `man sbatch` or https://slurm.schedmd.com/sbatch.html for descriptions of sbatch options.
#SBATCH --nodes=1                   # Number of nodes to request
#SBATCH --cpus-per-task=8           # Number of CPUs to request
#SBATCH --gpus=0                    # Number of GPUs to request
#SBATCH --mem=32G                   # Amount of RAM memory requested
#SBATCH -p ampere
##SBATCH --qos=standby

apptainer run --bind ./:/srv ../../apptainer/knn-join.sif "$@"

wait  # Wait for all jobs to complete
exit 0 # happy end
