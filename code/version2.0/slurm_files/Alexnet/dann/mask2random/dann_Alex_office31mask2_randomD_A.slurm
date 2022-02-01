#!/bin/bash

#SBATCH --job-name="ARM2DADANN"
#SBATCH --account punim1006
#SBATCH --qos="gpgpuresplat"
#SBATCH --partition gpgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --time=72:00:00
#SBATCH --mem=16G


#SBATCH -o Alexoffice31mask2RandomD_A_DANN.%N.%j.out # STDOUT 
#SBATCH -e Alexoffice31mask2RandomD_A_DANN.%N.%j.err # STDERR


#SBATCH --mail-type=ALL
#SBATCH --mail-user=junzhin@student.unimelb.edu.au


source activate base 

python3 ../../../../dann_modified.py /home/junzhin/Project/Summer_project/code/version2.0/data/office31 -j 4 -b 256 -d Office31 -s D -t A -a alexnet --epochs 30 --seed 1 --local --local-pretrained-path /home/junzhin/Project/Summer_project/code/version1.0/pretrained_model_results/AlEX_office31masked2Random_newmodel_best.pth.tar --log /home/junzhin/Project/Summer_project/code/version2.0/training_results/dann/ALEX_office31masked2Random/D_A
