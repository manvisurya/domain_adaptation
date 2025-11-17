#!/bin/bash -l
#
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=1-00:00:00
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

module load python
conda activate mmcv_env
export PYTHONPATH=$PYTHONPATH:/home/woody/iwnt/iwnt138h/dgp_dataset/dgp

# python train.py --model_name Virtual_gallery --dataset Virtual_gallery --width 704 --height 352 --data_path /home/woody/iwnt/iwnt138h/virtual_gallery_dataset --batch_size 8 --sup --learning_rate 1e-6
python train.py --model_name kitti_sup --dataset kitti --png --width 704 --height 352 --data_path /home/woody/iwnt/iwnt138h --batch_size 1 --resume --load_weights_folder /home/hpc/iwnt/iwnt138h/ada-depth/exp_logs/'Virtual_gallery(1e-6)'/models/weights_21