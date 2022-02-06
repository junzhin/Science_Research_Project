#!/bin/bash

#SBATCH --job-name="RROM2N"
#SBATCH --account punim1006
#SBATCH --qos="gpgpuresplat"
#SBATCH --partition gpgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --time=72:00:00
#SBATCH --mem=32G


#SBATCH -o Res18office31mask2Random_new.%N.%j.out # STDOUT 
#SBATCH -e Res18office31mask2Random_new.%N.%j.err # STDERR


#SBATCH --mail-type=ALL
#SBATCH --mail-user=junzhin@student.unimelb.edu.au


source activate base 
python3 ../pretrained_main_subset_random_choosen_experiment.py -a resnet18 -b 384 --epochs 100 --dist-url tcp://127.0.0.1:6900 --dist-backend nccl --multiprocessing-distributed -s -r -sp /home/junzhin/Project/Summer_project/code/version1.0/masked_office31_imagenetlabel2_manual_df.csv  --world-size 1 --rank 0  --log /home/junzhin/Project/Summer_project/code/version1.0/pretrained_model_results/Resnet18_office31masked2Random_new -atfn train_blurred  -avfn val_blurred -j 8 /data/gpfs/datasets/Imagenet


  
