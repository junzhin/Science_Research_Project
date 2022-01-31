#!/bin/bash

#SBATCH --job-name="AM1AD"
#SBATCH --account punim1006
#SBATCH --qos="gpgpuresplat"
#SBATCH --partition gpgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --time=72:00:00
#SBATCH --mem=16G


#SBATCH -o Alexoffice31mask1_A_D.%N.%j.out # STDOUT 
#SBATCH -e Alexoffice31mask1_A_D.%N.%j.err # STDERR


#SBATCH --mail-type=ALL
#SBATCH --mail-user=junzhin@student.unimelb.edu.au

source activate base 

python3 ../../../../dann_modified.py /home/junzhin/Project/Summer_project/code/version2.0/data/office31  -d Office31 -s A -t D -a resnet18 --epochs 30 --seed 1 --log 