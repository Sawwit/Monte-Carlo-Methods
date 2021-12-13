#!/bin/bash
#SBATCH --partition=hefstud
#SBATCH --output=std_%A_%a.txt
#SBATCH --mem=100M
#SBATCH --time=3-0:00:00
cd ~/monte-carlo-techniques-2021/Project
/software/anaconda3/bin/python3 HMC_scalar.py -w 4 -ki 0.08 -kf 0.18 -ka 2 -l 1.5 -f 'check'