#!/bin/bash

#SBATCH --job-name="R50WDP"
#SBATCH --account punim1006
#SBATCH --qos="gpgpuresplat"
#SBATCH --partition gpgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=16G


#SBATCH -o Res50office31W_D_Pseudo.%N.%j.out # STDOUT 
#SBATCH -e Res50office31W_D_Pseudo.%N.%j.err # STDERR


#SBATCH --mail-type=ALL
#SBATCH --mail-user=junzhin@student.unimelb.edu.au

source activate base 
python3 ../../main_SSL.py --dist_url tcp://127.0.0.1:10014 --multiprocessing_distributed --regular_only_feature --p_cutoff 0.95 --seed 1 --net resnet50 --epochs 30 --iters_per_epoch 1000  --source W --target D  --method PseudoLabel --save_dir /home/junzhin/Project/Summer_project/code/version3.0/training_results/self_training/Resnet50_Moco/W_D --local --local-pretrained-path /home/junzhin/Project/Summer_project/code/version1.0/pretrained_model_results/moco_v2_800ep_pretrain.pth.tar


python3 ../../main_SSL.py --dist_url tcp://127.0.0.1:10014 --multiprocessing_distributed --regular_only_feature --p_cutoff 0.95 --seed 2 --net resnet50 --epochs 30  --iters_per_epoch 1000  --source W --target D  --method PseudoLabel --save_dir /home/junzhin/Project/Summer_project/code/version3.0/training_results/self_training/Resnet50_Moco/W_D --local --local-pretrained-path /home/junzhin/Project/Summer_project/code/version1.0/pretrained_model_results/moco_v2_800ep_pretrain.pth.tar


python3 ../../main_SSL.py --dist_url tcp://127.0.0.1:10014 --multiprocessing_distributed --regular_only_feature --p_cutoff 0.95 --seed 3 --net resnet50 --epochs 30 --iters_per_epoch 1000  --source W --target D  --method PseudoLabel --save_dir /home/junzhin/Project/Summer_project/code/version3.0/training_results/self_training/Resnet50_Moco/W_D --local --local-pretrained-path /home/junzhin/Project/Summer_project/code/version1.0/pretrained_model_results/moco_v2_800ep_pretrain.pth.tar

 
 