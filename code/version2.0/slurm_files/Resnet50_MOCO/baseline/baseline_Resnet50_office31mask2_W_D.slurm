#!/bin/bash

#SBATCH --job-name="R50WDbase"
#SBATCH --account punim1006
#SBATCH --qos="gpgpuresplat"
#SBATCH --partition gpgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --time=48:00:00
#SBATCH --mem=16G


#SBATCH -o Res50office31mask2W_D_baseline.%N.%j.out # STDOUT 
#SBATCH -e Res50office31mask2W_D_baseline.%N.%j.err # STDERR


#SBATCH --mail-type=ALL
#SBATCH --mail-user=junzhin@student.unimelb.edu.au

source activate base 
python3 ../../../source_only.py /home/junzhin/Project/Summer_project/code/version2.0/data/office31 -j 4 --seed 1 -d Office31 -s W -t D -a resnet50 --epochs 30  --local --local-pretrained-path /home/junzhin/Project/Summer_project/code/version1.0/pretrained_model_results/moco_v2_800ep_pretrain.pth.tar --log /home/junzhin/Project/Summer_project/code/version2.0/training_results/baseline/Resnet50_office31/W_D

python3 ../../../source_only.py /home/junzhin/Project/Summer_project/code/version2.0/data/office31 -j 4 --seed 2 -d Office31 -s W -t D -a resnet50 --epochs 30  --local --local-pretrained-path /home/junzhin/Project/Summer_project/code/version1.0/pretrained_model_results/moco_v2_800ep_pretrain.pth.tar --log /home/junzhin/Project/Summer_project/code/version2.0/training_results/baseline/Resnet50_office31/W_D

python3 ../../../source_only.py /home/junzhin/Project/Summer_project/code/version2.0/data/office31 -j 4 --seed 3 -d Office31 -s W -t D -a resnet50 --epochs 30  --local --local-pretrained-path /home/junzhin/Project/Summer_project/code/version1.0/pretrained_model_results/moco_v2_800ep_pretrain.pth.tar --log /home/junzhin/Project/Summer_project/code/version2.0/training_results/baseline/Resnet50_office31/W_D