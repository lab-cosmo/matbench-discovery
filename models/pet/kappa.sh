#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=1
#SBATCH -p h100,l40s
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task 1
#SBATCH --mem 50G
#SBATCH --time 3-0

source /home/bigi/virtualenv-k/bin/activate

echo STARTING AT `date`
python -u test_pet_kappa.py
echo FINISHED at `date`
