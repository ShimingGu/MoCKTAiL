#!/bin/bash -l

#SBATCH -n 64
#SBATCH -J 2PCF
#SBATCH -o res2pcf
#SBATCH -e err2pcf
#SBATCH -p cosma
#SBATCH -A durham
#SBATCH --exclusive
#SBATCH -t 72:00:00

# source /etc/profile.d/modules.csh
# module load python/2.7.13

module purge
module load python/3.6.5

# acd /cosma/home/durham/ddsm49/Py2PCF/
cd /cosma/home/durham/ddsm49/mocktail/github/MoCKTAiL
python3 Main.py

