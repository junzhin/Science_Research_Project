#!/bin/bash

#SBATCH --job-name="ROFULLN"
#SBATCH --account punim1006
#SBATCH --qos="gpgpuresplat"
#SBATCH --partition gpgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16  
#SBATCH --gres=gpu:4
#SBATCH --time=72:00:00
#SBATCH --mem=16G


#SBATCH -o Resnet18_office31full.%N.%j.out # STDOUT 
#SBATCH -e Resnet18_office31full.%N.%j.err # STDERR


#SBATCH --mail-type=ALL
#SBATCH --mail-user=junzhin@student.unimelb.edu.au


source activate base 
python3 ../pretrained_main_subset_random_choosen_experiment.py -a resnet18 -b 512 --epochs 100 --dist-url tcp://127.0.0.1:8001 --dist-backend nccl --multiprocessing-distribute --world-size 1 --rank 0 --log /home/junzhin/Project/Summer_project/code/version1.0/pretrained_model_results/Resnet18_office31full -atfn train_blurred -avfn val_blurred -j 16 /data/gpfs/datasets/Imagenet


 