#!/bin/bash

#SBATCH --partition=a100
#SBATCH --account=jinlab
#SBATCH --gres=gpu:a100:2
#SBATCH --time=72:00:00
conda activate XD
python train_final.py
