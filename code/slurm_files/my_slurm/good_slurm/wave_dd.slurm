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

#python -u /people/beec613/pnnl_research/code/damiens_code/good_code/wave_dd/onet_scripts/MF_script_loop.py 0 0.6 
#python -u /people/beec613/pnnl_research/code/damiens_code/good_code/wave_dd/onet_scripts/MF_script_loop.py 0 0.8
#python -u /people/beec613/pnnl_research/code/damiens_code/good_code/wave_dd/onet_scripts/MF_script_loop.py 0 1
#python -u /people/beec613/pnnl_research/code/damiens_code/good_code/wave_dd/onet_scripts/MF_script_loop.py 0 1.2
#python -u /people/beec613/pnnl_research/code/damiens_code/good_code/wave_dd/onet_scripts/MF_script_loop.py 0 1.4

#python -u /people/beec613/pnnl_research/code/damiens_code/good_code/wave_dd/onet_scripts/MF_script_loop.py 0 1 0.01 0.95 80
#python -u /people/beec613/pnnl_research/code/damiens_code/good_code/wave_dd/onet_scripts/MF_script_loop.py 0 1 0.001 0.95 80
#python -u /people/beec613/pnnl_research/code/damiens_code/good_code/wave_dd/onet_scripts/MF_script_loop.py 0 1 0.0001 0.95 80
#python -u /people/beec613/pnnl_research/code/damiens_code/good_code/wave_dd/onet_scripts/MF_script_loop.py 0 1 0.01 0.99 80
#python -u /people/beec613/pnnl_research/code/damiens_code/good_code/wave_dd/onet_scripts/MF_script_loop.py 0 1 0.001 0.99 80
#python -u /people/beec613/pnnl_research/code/damiens_code/good_code/wave_dd/onet_scripts/MF_script_loop.py 0 1 0.0001 0.99 80
#python -u /people/beec613/pnnl_research/code/damiens_code/good_code/wave_dd/onet_scripts/MF_script_loop.py 0 1 0.01 0.95 60
#python -u /people/beec613/pnnl_research/code/damiens_code/good_code/wave_dd/onet_scripts/MF_script_loop.py 0 1 0.001 0.95 60
python -u /people/beec613/pnnl_research/code/damiens_code/good_code/wave_dd/onet_scripts/MF_script_loop.py 0 1 0.0001 0.95 60
python -u /people/beec613/pnnl_research/code/damiens_code/good_code/wave_dd/onet_scripts/MF_script_loop.py 0 1 0.01 0.99 60
python -u /people/beec613/pnnl_research/code/damiens_code/good_code/wave_dd/onet_scripts/MF_script_loop.py 0 1 0.001 0.99 60
python -u /people/beec613/pnnl_research/code/damiens_code/good_code/wave_dd/onet_scripts/MF_script_loop.py 0 1 0.0001 0.99 60

