#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --ntasks-per-node 1
#SBATCH --gpus-per-node 1
#SBATCH --cpus-per-task 8
#SBATCH --mem=12G
#SBATCH --time=1:00:00

module load gcc python
source /home/bigi/virtualenv-i/bin/activate

mtt --debug train options.yaml
