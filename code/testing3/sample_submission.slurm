#!/bin/bash
#SBATCH --mail-user=junzhin@student.unimelb.edu.au
#SBATCH --mail-type=ALL
module load foss/2019b
module load python/3.7.4
srun /home/junzhin/testing/testing.py
