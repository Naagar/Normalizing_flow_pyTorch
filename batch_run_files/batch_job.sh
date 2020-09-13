#!/bin/bash
#SBATCH -A research
#SBATCH --qos=medium
#SBATCH -n 40
#SBATCH --gres=gpu:0
#SBATCH --mem-per-cpu=2048
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=END

module load python/3.6.8
cd "Auto. Navigation"
python CNN_with_backprop.py