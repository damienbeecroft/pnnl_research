#!/bin/bash
#SBATCH -p dl
#SBATCH --gres=gpu:2
#SBATCH --job-name=pendulum

#Run code
python test.py
