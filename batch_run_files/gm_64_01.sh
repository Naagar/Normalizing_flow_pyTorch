#!/bin/bash              
#SBATCH --job-name=imnet6_em                # Job name                        
#SBATCH --time=4-0:0:0  
#SBATCH --output=log-%j.out       # Output file.                    
#SBATCH --gres=gpu:2       # N number of GPU devices.            
#SBATCH --mail-type=END                # Enable email                    
#SBATCH --mail-user=sandeep.nagar@research.iiit.ac.in    # Where to send mail  
#SBATCH --mem=64G         
###SBATCH --nodelist=gnode15
####SBATCH --ntasks=2
# Enter command here  





####SBATCH --ntasks=4
# Enter command here    
#yes y |ssh-keygen -q -t rsa -N '' >/dev/null
#ssh-copy-id Destname@dest_IP
#rsync -zaPq /ssd_scratch/cvit/file1  Destname@dest_IP:/destination_file/

mkdir -p /ssd_scratch/cvit/girish.varma/dataset/
rsync -zaPq girish.varma@ada.iiit.ac.in:/share1/dataset/ImageNet-2015 /ssd_scratch/cvit/girish.varma/dataset

#cd /ssd_scratch/cvit/girish.varma/Normalizing-flow-master
export LD_LIBRARY_PATH=/usr/local/apps/cuDNN/7-cuda10/lib64/
source san_env/bin/activate

mpiexec -n 2 python3 train.py --problem imagenet-oord --image_size 64 --n_level 4 --depth 48 --flow_permutation 7 --flow_coupling 1 --seed 0 --learnprior --lr 0.001 --n_bits_x 8 --data_dir /ssd_scratch/cvit/girish_san/dataset


