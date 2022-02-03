#!/bin/bash

#SBATCH --job-name="Res18Mask1"
#SBATCH --account punim1006
#SBATCH --qos="gpgpuresplat"
#SBATCH --partition gpgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --time=96:00:00
#SBATCH --mem=16G


#SBATCH -o Res18office31mask1.%N.%j.out # STDOUT 
#SBATCH -e Res18office31mask1.%N.%j.err # STDERR


#SBATCH --mail-type=ALL
#SBATCH --mail-user=junzhin@student.unimelb.edu.au

source activate base
python3 ../pretrained_main_subset.py -a resnet18 -b 128 --epochs 50 --dist-url tcp://127.0.0.1:3000 --dist-backend nccl --multiprocessing-distributed -s -sp /home/junzhin/Project/Summer_project/code/version1.0/masked_office31_imagenetlabel1_df.csv --world-size 1 --rank 0 --resume /home/junzhin/Project/Summer_project/code/version1.0/pretrained_model_results/Resnet18_office31masked1checkpoint.pth.tar  --log /home/junzhin/Project/Summer_project/code/version1.0/pretrained_model_results/Resnet18_office31masked1  -atfn train_blurred  -avfn val_blurred -j 4 /data/gpfs/datasets/Imagenet