#!/bin/bash

#SBATCH --job-name="RM1ADSSL"
#SBATCH --account punim1006
#SBATCH --qos="gpgpuresplat"
#SBATCH --partition gpgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --time=72:00:00
#SBATCH --mem=16G


#SBATCH -o Res18office31mask1_A_DSSL.%N.%j.out # STDOUT 
#SBATCH -e Res18office31mask1_A_DSSL.%N.%j.err # STDERR


#SBATCH --mail-type=ALL
#SBATCH --mail-user=junzhin@student.unimelb.edu.au

source activate base 

python3 ../../../../main_SSL.py --dist_url tcp://127.0.0.1:10035 --multiprocessing_distributed --regular_only_feature --p_cutoff 0.95 --seed 1  --epochs 30  --batchsize 128 --iters_per_epoch 250  --source A --target D  --method PseudoLabel --save_dir /home/junzhin/Project/Summer_project/code/version3.0/training_results/self_training/Resnet18_office31masked1/A_D --local --local-pretrained-path /home/junzhin/Project/Summer_project/code/version1.0/pretrained_model_results/Resnet18_office31masked1_newmodel_best.pth.tar