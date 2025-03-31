#!/bin/bash
#SBATCH --job-name=unet_train
#SBATCH --output=logs/unet_train_%j.out
#SBATCH --error=logs/unet_train_%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --begin=19:00  # <-- start at 7 PM German time

# Load environment
source ~/.bashrc
conda activate unet-env  # Change to your actual environment

# Run training
python train_unet.py --epochs 50