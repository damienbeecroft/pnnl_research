#!/bin/bash
#SBATCH -t 20:00:00
#SBATCH -J t1
#SBATCH --nodes=1
#SBATCH --tasks-per-node=8
##SBATCH --mem=32g

#SBATCH -A sea-crogs
#SBATCH -p dl_shared
#SBATCH --gres=gpu:1

#SBATCH -o "tfjob1-%j.out"
#SBATCH -e "tfjob1-%j.err"


module purge
module load cuda/11.4
module load python/miniconda3.9
source /share/apps/python/miniconda3.9/etc/profile.d/conda.sh
conda activate jax-cuda


# python -u ./SF_script.py
python /people/beec613/pnnl_research/code/amandas_code/onet_scripts/train_MF_EWC_script.py
