#!/bin/bash
       #SBATCH -p physical
       #SBATCH --time=01:00:00
       #SBATCH --nodes=1
       #SBATCH --ntasks=1
       #SBATCH --cpus-per-task=1
       module load my-app-compiler/version
       my-app data
       #SBATCH --mail-user=junzhin@student.unimelb.edu.au
       #SBATCH --mail-type=ALL

       #!/bin/bash
       #SBATCH --ntasks=4
       module load foss/2019b
       module load python/3.7.4
        srun -n 4 /home/junzhin/testing/testing.py
