#!/bin/bash
#SBATCH -p dl
#SBATCH --gres=gpu:2
#SBATCH --job-name=pendulum

# Damiens code
# module purge
# module load python/3.7.0 
# module load jax

# Amandas code
module purge
module load cuda/11.4
module load python/miniconda3.9
source /share/apps/python/miniconda3.9/etc/profile.d/conda.sh
conda create –-name jax-cuda
conda activate jax-cuda

conda install scipy=1.10.0
pip install --upgrade pip
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
conda install tqdm
# conda install matplotlib
pip install torch
conda install cudnn

python /people/beec613/pnnl_research/code/amandas_code/onet_scripts/train_MF_EWC_script.py
