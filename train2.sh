#!/bin/bash -l
#
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=1-00:00:00
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

module load python
conda activate mmcv_env

python nyuadapt.py --dataset nyu --log_dir ./adapt_logs --num_workers 4 --batch_size 1 --learning_rate 1e-4 --num_epochs 1