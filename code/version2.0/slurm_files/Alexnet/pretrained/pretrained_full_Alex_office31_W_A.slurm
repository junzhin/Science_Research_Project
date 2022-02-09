#!/bin/bash

#SBATCH --job-name="AM2WAbasefull"
#SBATCH --account punim1006
#SBATCH --qos="gpgpuresplat"
#SBATCH --partition gpgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --time=72:00:00
#SBATCH --mem=16G


#SBATCH -o Alexoffice31mask2W_A_pretrained.%N.%j.out # STDOUT 
#SBATCH -e Alexoffice31mask2W_A_pretrained.%N.%j.err # STDERR


#SBATCH --mail-type=ALL
#SBATCH --mail-user=junzhin@student.unimelb.edu.au


source activate base 
python3 ../../../source_only.py /home/junzhin/Project/Summer_project/code/version2.0/data/office31 -j 4 -b 256 -d Office31 -s W -t A -a alexnet --epochs 30  --log /home/junzhin/Project/Summer_project/code/version2.0/training_results/Alex_pretrained/W_A