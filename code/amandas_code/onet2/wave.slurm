#!/bin/bash
#SBATCH -t 20:00:00
#SBATCH -J t1
#SBATCH --nodes=1

#SBATCH -A sea-crogs
#SBATCH -p dl

#SBATCH -o "tfjob1-%j.out"
#SBATCH -e "tfjob1-%j.err"

module purge
module load python/miniconda3.9
source /share/apps/python/miniconda3.9/etc/profile.d/conda.sh
conda activate jax-gpu


python -u MF_script_loop.py
