#!/bin/bash
#SBATCH -A research
#SBATCH --job-name=___
#SBATCH --qos=medium
#SBATCH -n 30
#SBATCH --gres=gpu:3
#SBATCH --output=output_log_files/Inv_Conv_logQ%j.out       # Output file.
#SBATCH --mail-type=END                # Enable email about job finish 
#SBATCH --mail-user=sandeep.nagar@research.iiit.ac.in    # Where to send mail  
#SBATCH --mem-per-cpu=3000
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=END

module load cudnn/7-cuda-10.0

#source venv/bin/activate
# cd Glow_pyTorch/glow/
mpiexec -n 3 python3 train_1.py --num_epochs 300 --num_channels 256 --num_steps 32 --warm_up 500000 --batch_size 8