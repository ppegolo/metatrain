#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks 2
#SBATCH --ntasks-per-node 2
#SBATCH --gpus-per-node 2
#SBATCH --cpus-per-task 8
#SBATCH --exclusive
#SBATCH --time=1:00:00

module load gcc python
source /home/bigi/virtualenv-i/bin/activate

srun mtt --debug train options-distributed.yaml
