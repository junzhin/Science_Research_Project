#!/bin/bash

#SBATCH --job-name="AM2WDDANN"
#SBATCH --account punim1006
#SBATCH --qos="gpgpuresplat"
#SBATCH --partition gpgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=16G


#SBATCH -o Alexoffice31mask2W_D_DANN.%N.%j.out # STDOUT 
#SBATCH -e Alexoffice31mask2W_D_DANN.%N.%j.err # STDERR


#SBATCH --mail-type=ALL
#SBATCH --mail-user=junzhin@student.unimelb.edu.au

source activate base 
python3 ../../../../dann_modified.py /home/junzhin/Project/Summer_project/code/version2.0/data/office31 -j 8 -d Office31 -s W -t D -a alexnet_cdan --epochs 50  --seed 1  --local --local-pretrained-path /home/junzhin/Project/Summer_project/code/version1.0/pretrained_model_results/ALEX_office31masked2_CDANmodel_best.pth.tar --log /home/junzhin/Project/Summer_project/code/version2.0/training_results/dann/ALEX_office31masked2/W_D