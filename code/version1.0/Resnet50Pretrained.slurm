#!/bin/bash

#SBATCH --job-name="Resnet50Tinydataset"
#SBATCH --account punim1006
#SBATCH --qos="gpgpuresplat"
#SBATCH --partition gpgpu
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:4
#SBATCH --time=0-24:00:00


#SBATCH -o slurm.%N.%j.out # STDOUT 
#SBATCH -e slurm.%N.%j.err # STDERR


#SBATCH --mail-type=ALL
#SBATCH --mail-user=junzhin@student.unimelb.edu.au


source activate base 


python3 main.py -a resnet50 -b 512  --dist-url tcp://127.0.0.1:8080 --dist-backend nccl --rank 0 --world-size 1 --multiprocessing-distributed  /home/junzhin/Project/datasets/tiny-imagenet-200

