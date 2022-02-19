#!/bin/bash

#SBATCH --job-name="AM2AWbase"
#SBATCH --account punim1006
#SBATCH --qos="gpgpuresplat"
#SBATCH --partition gpgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=32G


#SBATCH -o Alexoffice31mask2A_W_baseline.%N.%j.out # STDOUT 
#SBATCH -e Alexoffice31mask2A_W_baseline.%N.%j.err # STDERR


#SBATCH --mail-type=ALL
#SBATCH --mail-user=junzhin@student.unimelb.edu.au
source activate base 
python3 ../../../../source_only.py /home/junzhin/Project/Summer_project/code/version2.0/data/office31 -j 8 -d Office31 -s A -t W -a alexnet_cdan --epochs 50 --seed 1 --local --local-pretrained-path /home/junzhin/Project/Summer_project/code/version1.0/pretrained_model_results/ALEX_office31masked2_CDANmodel_best.pth.tar --log /home/junzhin/Project/Summer_project/code/version2.0/training_results/baseline/ALEX_office31masked2/A_W