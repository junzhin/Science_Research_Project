#!/bin/bash

#SBATCH --job-name="Resnet50"
#SBATCH --account punim1006
#SBATCH --qos="gpgpuresplat"
#SBATCH --partition gpgpu
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH --time=0-72:00:00


#SBATCH -o slurm.%N.%j.out # STDOUT 
#SBATCH -e slurm.%N.%j.err # STDERR


#SBATCH --mail-type=ALL
#SBATCH --mail-user=junzhin@student.unimelb.edu.au


source activate base 

python3 pretrained_main.py -a resnet50 -b 512  --dist-url tcp://127.0.0.1:8080 --dist-backend nccl --rank 0 --world-size 1 --epochs 85 --multiprocessing-distributed --resume /home/junzhin/Project/Summer_project/code/version1.0/pretrained_model_results/Resnet50checkpoint.pth.tar  --log pretrained_model_results/Resnet50 -atfn train_blurred  -avfn val_blurred /data/gpfs/datasets/Imagenet


 