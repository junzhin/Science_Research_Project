#!/bin/bash

#SBATCH --job-name="ADWbasefull"
#SBATCH --account punim1006
#SBATCH --qos="gpgpuresplat"
#SBATCH --partition gpgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --mem=32G


#SBATCH -o Alexoffice31D_W_pretrained.%N.%j.out # STDOUT 
#SBATCH -e Alexoffice31D_W_pretrained.%N.%j.err # STDERR


#SBATCH --mail-type=ALL
#SBATCH --mail-user=junzhin@student.unimelb.edu.au

source activate base 
python3 ../../../source_only.py /home/junzhin/Project/Summer_project/code/version2.0/data/office31 -j 8 --seed 1 -d Office31 -s D -t W -a alexnet --epochs 30 --log /home/junzhin/Project/Summer_project/code/version2.0/training_results/Alex_pretrained/source_only/D_A

python3 ../../../source_only.py /home/junzhin/Project/Summer_project/code/version2.0/data/office31 -j 8 --seed 2 -d Office31 -s D -t W -a alexnet --epochs 30 --log /home/junzhin/Project/Summer_project/code/version2.0/training_results/Alex_pretrained/source_only/D_A

python3 ../../../source_only.py /home/junzhin/Project/Summer_project/code/version2.0/data/office31 -j 8 --seed 3 -d Office31 -s D -t W -a alexnet --epochs 30 --log /home/junzhin/Project/Summer_project/code/version2.0/training_results/Alex_pretrained/source_only/D_A