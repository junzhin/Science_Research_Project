#!/bin/bash

#SBATCH --job-name="Alexmask2"
#SBATCH --account punim1006
#SBATCH --qos="gpgpuresplat"
#SBATCH --partition gpgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --time=96:00:00
#SBATCH --mem=16G


#SBATCH -o Alexoffice31mask2.%N.%j.out # STDOUT 
#SBATCH -e Alexoffice31mask2.%N.%j.err # STDERR


#SBATCH --mail-type=ALL
#SBATCH --mail-user=junzhin@student.unimelb.edu.au


source activate base 
python3 ../pretrained_main_subset.py -a alexnet --lr 0.01 --epochs 100 --dist-url tcp://127.0.0.1:8080 --dist-backend nccl --multiprocessing-distributed -s -sp /home/junzhin/Project/Summer_project/code/version1.0/masked_office31_imagenetlabel2_manual_df.csv  --resume /home/junzhin/Project/Summer_project/code/version1.0/pretrained_model_results/ALEX_office31masked2checkpoint.pth.tar --world-size 1 --rank 0  --log /home/junzhin/Project/Summer_project/code/version1.0/pretrained_model_results/ALEX_office31masked2 -atfn train_blurred  -avfn val_blurred -j 4 /data/gpfs/datasets/Imagenet


