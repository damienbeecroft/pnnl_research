#!/bin/bash
#SBATCH -t 20:00:00
#SBATCH -J t1
#SBATCH --nodes=1
#SBATCH --tasks-per-node=8
#SBATCH --mail-type=END
#SBATCH --mail-user=damien.beecroft@pnnl.gov
##SBATCH --mem=32g

#SBATCH -A sea-crogs
#SBATCH -p dl
#SBATCH --gres=gpu:1

#SBATCH -o "tfjob1-%j.out"
#SBATCH -e "tfjob1-%j.err"


module purge
# module load cuda/11.4
module load python/miniconda3.9
source /share/apps/python/miniconda3.9/etc/profile.d/conda.sh
conda activate jax-gpu 
# :os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"

#python -u /people/beec613/pnnl_research/code/damiens_code/Pendulum_DD/Pendulum_DD/onet_scripts/train_MF_EWC_script.py 0 20
python -u /people/beec613/pnnl_research/code/damiens_code/Pendulum_DD/Pendulum_DD/onet_scripts/train_MF_EWC_script.py 0 30
python -u /people/beec613/pnnl_research/code/damiens_code/Pendulum_DD/Pendulum_DD/onet_scripts/train_MF_EWC_script.py 0 40 
python -u /people/beec613/pnnl_research/code/damiens_code/Pendulum_DD/Pendulum_DD/onet_scripts/train_MF_EWC_script.py 0 70
python -u /people/beec613/pnnl_research/code/damiens_code/Pendulum_DD/Pendulum_DD/onet_scripts/train_MF_EWC_script.py 0 100
