#!/bin/bash

#SBATCH --job-name="R50AWbase"
#SBATCH --account punim1006
#SBATCH --qos="gpgpuresplat"
#SBATCH --partition gpgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=16G


#SBATCH -o Res50office31mask2A_W_baseline.%N.%j.out # STDOUT 
#SBATCH -e Res50office31mask2A_W_baesline.%N.%j.err # STDERR


#SBATCH --mail-type=ALL
#SBATCH --mail-user=junzhin@student.unimelb.edu.au

source activate base 
python3 ../../../source_only.py /home/junzhin/Project/Summer_project/code/version2.0/data/office31 -j 4  -d Office31 -s A -t W -a resnet50 --epochs 30 --seed 1 --local --local-pretrained-path /home/junzhin/Project/Summer_project/code/version1.0/pretrained_model_results/moco_v2_800ep_pretrain.pth.tar --log /home/junzhin/Project/Summer_project/code/version2.0/training_results/baseline/Resnet50_office31/A_W

python3 ../../../source_only.py /home/junzhin/Project/Summer_project/code/version2.0/data/office31 -j 4  -d Office31 -s A -t W -a resnet50 --epochs 30 --seed 2 --local --local-pretrained-path /home/junzhin/Project/Summer_project/code/version1.0/pretrained_model_results/moco_v2_800ep_pretrain.pth.tar --log /home/junzhin/Project/Summer_project/code/version2.0/training_results/baseline/Resnet50_office31/A_W

python3 ../../../source_only.py /home/junzhin/Project/Summer_project/code/version2.0/data/office31 -j 4  -d Office31 -s A -t W -a resnet50 --epochs 30 --seed 3 --local --local-pretrained-path /home/junzhin/Project/Summer_project/code/version1.0/pretrained_model_results/moco_v2_800ep_pretrain.pth.tar --log /home/junzhin/Project/Summer_project/code/version2.0/training_results/baseline/Resnet50_office31/A_W