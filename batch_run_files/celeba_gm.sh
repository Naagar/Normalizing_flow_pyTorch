#!/bin/bash                                                               
#SBATCH --job-name=cel72                # Job name                        
#SBATCH --time=7-0:0:0        
#SBATCH --output=Output_files/Zceleb71-%j.out       # Output file.    
#SBATCH -n 40                
#SBATCH --gres=gpu:4       # N number of GPU devices.            
#SBATCH --mail-type=END                # Enable email                    
#SBATCH --mail-user=sandeep.nagar@research.iiit.ac.in    # Where to send mail  
##SBATCH --mem=120G  
##SBATCH --nodelist=gnode41
# Enter command here

mkdir -p /ssd_scratch/cvit/girish.varma/celeba_dataset

mkdir -p /ssd_scratch/cvit/girish.varma/Log_celeba

rsync -aPq girish.varma@ada.iiit.ac.in:/share1/girish.varma/celeba-tfr  /ssd_scratch/cvit/girish.varma/celeba_dataset/


export LD_LIBRARY_PATH=/usr/local/apps/cuDNN/7-cuda10/lib64/
source san_env/bin/activate

# mpiexec -n 2 python3 train.py --problem celeba --image_size 256 --n_level 6 --depth 16 --flow_permutation 7 --flow_coupling 2 --seed 2 --learnprior --lr 0.005 --n_bits_x 5 --epochs 1001 --data_dir /ssd_scratch/cvit/girish.varma/emerging/celeba-tfr --logdir ./log_celeb --restore --epochs_warmup 1

mpiexec -n 4 python3 train_test.py --problem celeba --image_size 256 --n_level 4 --depth 16 --flow_permutation 7 --flow_coupling 1 --seed 2 --learnprior --n_bits_x 5 --data_dir ssd_scratch/cvit/girish.varma/celeba_dataset/celeba-tfr  --epochs 600 --restore --infer


###############