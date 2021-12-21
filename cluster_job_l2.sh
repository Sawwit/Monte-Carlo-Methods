#!/bin/bash
#SBATCH --partition=hefstud
#SBATCH --output=std_%A_%a.txt
#SBATCH --mem=100M
#SBATCH --time=3-0:00:00
#SBATCH --array=0-13%3
cd ~/monte-carlo-techniques-2021/Project
lattices=($LANG=en_US seq 3 1 17)
width=${lattices[$SLURM_ARRAY_TASK_ID]}
/software/anaconda3/bin/python3 cluster.py -w ${width} -ki 0.08 -kf 0.18 -ka 11 -l 2