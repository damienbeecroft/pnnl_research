#!/bin/bash
#SBATCH -p dl
#SBATCH --gres=gpu:2
#SBATCH --job-name=pendulum

#Run code
module load scipy.io 
python /people/beec613/pnnl_research/code/amandas_code/onet_scripts/onet_scripts/train_MF_EWC_script.py
