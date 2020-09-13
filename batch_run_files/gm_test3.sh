#!/bin/bash                                 
#SBATCH --job-name=cifar72                # Job name     
#SBATCH --time=6-0:0:0        
#SBATCH --output=log-emerging-%j.out       # Output file.                    
#SBATCH --gres=gpu:2       # N number of GPU devices.            
#SBATCH --mail-type=NONE                # Enable email                    
#SBATCH --mail-user=sandeep.nagar@research.iiit.ac.in    # Where to send mail  
#SBATCH --mem=40G         
#SBATCH --nodelist=gnode15
####SBATCH --ntasks=2
# Enter command here    

cd /ssd_scratch/cvit/girish.varma/Normalizing-flow-master
export LD_LIBRARY_PATH=/usr/local/apps/cuDNN/7-cuda10/lib64/
source venv/bin/activate

mpiexec -n 2 python3 train.py --problem cifar10 --image_size 32 --n_level 3 --depth 32 --flow_permutation 7 --flow_coupling 2 --seed 2 --learnprior --lr 0.001 --n_bits_x 8 --epochs 4001 --logdir ./log_cif72 --restore
####################