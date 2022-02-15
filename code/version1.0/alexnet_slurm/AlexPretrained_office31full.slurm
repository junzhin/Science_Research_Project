#!/bin/bash

#SBATCH --job-name="A2FULLN"
#SBATCH --account punim1006
#SBATCH --qos="gpgpuresplat"
#SBATCH --partition gpgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --time=48:00:00
#SBATCH --mem=32G


#SBATCH -o Alexoffice31full2.%N.%j.out # STDOUT 
#SBATCH -e Alexoffice31full2.%N.%j.err # STDERR


#SBATCH --mail-type=ALL
#SBATCH --mail-user=junzhin@student.unimelb.edu.au



source activate base 
python3 ../pretrained_main_subset_random_choosen_experiment_CDAN.py -a alexnet_cdan --lr 0.01 -b 256 --epochs 100 --dist-url tcp://127.0.0.1:7001 --dist-backend nccl --multiprocessing-distributed --resume  /home/junzhin/Project/Summer_project/code/version1.0/pretrained_model_results/AlEX_office31full2checkpoint.pth.tar --world-size 1 --rank 0  --log /home/junzhin/Project/Summer_project/code/version1.0/pretrained_model_results/AlEX_office31full2 -atfn train_blurred  -avfn val_blurred -j 8 /data/gpfs/datasets/Imagenet