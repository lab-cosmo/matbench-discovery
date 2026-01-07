#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=1
#SBATCH -p h100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task 1
#SBATCH --mem 50G
#SBATCH --time 3-0
#SBATCH --array=0-7

source /home/bigi/virtualenv-k/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo start at `date`

disparray=(0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1)

python test_pet_orb_kappa.py $1 ${disparray[$SLURM_ARRAY_TASK_ID]}
echo end at `date`
