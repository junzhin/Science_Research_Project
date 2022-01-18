#!/bin/bash

#SBATCH --job-name="Resnet50Tinydataset"
#SBATCH --account punim1006
#SBATCH --qos="gpgpuresplat"
#SBATCH --partition gpgpu

#SBATCH --nodes=2                    # Maximum number of nodes to be allocated
#SBATCH --ntasks=2 
#SBATCH --ntasks-per-node=1         # Maximum number of tasks on each node
#SBATCH --cpus-per-task=4
#SBATCH --time=0-01:00:00
#SBATCH --gres=gpu:4
#SBATCH --hint=nomultithread

 
#SBATCH -o slurm.%N.%j.out # STDOUT 
#SBATCH -e slurm.%N.%j.err # STDERR


#SBATCH --mail-type=ALL
#SBATCH --mail-user=junzhin@student.unimelb.edu.au

source activate base 

srun --ntasks=1  python3 pretrained_main.py -a resnet50 -b 512  --dist-url tcp://127.0.0.1:8080 --dist-backend nccl --rank 0 --world-size 2 --epochs 100 --multiprocessing-distributed /home/junzhin/Project/datasets/tiny-imagenet-200 &

srun --ntasks=1  python3 pretrained_main.py -a resnet50 -b 512  --dist-url tcp://127.0.0.1:8080 --dist-backend nccl --rank 1 --world-size 2 --epochs 100 --multiprocessing-distributed --log pretrained_model_results/Tiny_imagenet /home/junzhin/Project/datasets/tiny-imagenet-200 &

wait

 