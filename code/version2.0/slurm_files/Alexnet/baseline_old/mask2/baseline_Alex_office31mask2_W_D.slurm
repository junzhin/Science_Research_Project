#!/bin/bash

#SBATCH --job-name="AM2WDbase"
#SBATCH --account punim1006
#SBATCH --qos="gpgpuresplat"
#SBATCH --partition gpgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --time=72:00:00
#SBATCH --mem=16G


#SBATCH -o Alexoffice31mask2W_baseline.%N.%j.out # STDOUT 
#SBATCH -e Alexoffice31mask2W_baesline.%N.%j.err # STDERR


#SBATCH --mail-type=ALL
#SBATCH --mail-user=junzhin@student.unimelb.edu.au

source activate base 
python3 ../../../../source_only.py /home/junzhin/Project/Summer_project/code/version2.0/data/office31 -j 4 -b 256 -d Office31 -s W -t D -a alexnet --epochs 30  --local --local-pretrained-path /home/junzhin/Project/Summer_project/code/version1.0/old_model_results/ALEX_office31masked2_newmodel_best.pth.tar --log /home/junzhin/Project/Summer_project/code/version2.0/training_results/baseline_old/ALEX_office31masked2/W_D
