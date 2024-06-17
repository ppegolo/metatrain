#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks 2
#SBATCH --ntasks-per-node 2
#SBATCH --gpus-per-node 2
#SBATCH --cpus-per-task 8
#SBATCH --time=1:00:00

# Activate your modules, environments, containers etc here

srun mtt train options-distributed.yaml
