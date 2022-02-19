#!/bin/bash

#SBATCH --job-name="AM2WASSL"
#SBATCH --account punim1006
#SBATCH --qos="gpgpuresplat"
#SBATCH --partition gpgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=16G


#SBATCH -o Alexoffice31mask2W_ASSL.%N.%j.out # STDOUT 
#SBATCH -e Alexoffice31mask2W_ASSL.%N.%j.err # STDERR


#SBATCH --mail-type=ALL
#SBATCH --mail-user=junzhin@student.unimelb.edu.au


source activate base 

python3 ../../../../main_SSL.py --dist_url tcp://127.0.0.1:10028 --multiprocessing_distributed --regular_only_feature --net alexnet_cdan --p_cutoff 0.95 --seed 1  --epochs 50  --batchsize 128 --iters_per_epoch 500  --source W --target A  --method PseudoLabel --save_dir /home/junzhin/Project/Summer_project/code/version3.0/training_results/self_training/ALEX_office31masked2/W_A --local --local-pretrained-path /home/junzhin/Project/Summer_project/code/version1.0/pretrained_model_results/ALEX_office31masked2_CDANmodel_best.pth.tar