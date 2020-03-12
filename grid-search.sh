#!/bin/bash

#SBATCH --job-name=finalProject
#SBATCH --output=finalProject_%A_%a.out
#SBATCH --error=finalProject_%A_%a.err
#SBATCH --array=1-30
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=45

# Print the task id.
echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID

# Add lines here to run your computations.
python random_search.py $SLURM_ARRAY_TASK_ID
