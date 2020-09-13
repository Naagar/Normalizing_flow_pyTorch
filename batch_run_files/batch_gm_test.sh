#!/bin/bash
#SBATCH -A research
#SBATCH --qos=medium
#SBATCH -n 40
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=2048
#SBATCH --time=0-00:03:00
#SBATCH --mail-type=END

module load cudnn/7-cuda-10.0
source venv/bin/activate
cd Auto.\ Navigation/gen_models/Normalizing-flow-master/
python3 test_1.py

