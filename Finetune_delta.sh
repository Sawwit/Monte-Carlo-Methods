#!/bin/bash
#SBATCH --partition=hefstud
#SBATCH --output=std_%A_%a.txt
#SBATCH --mem=100M
#SBATCH --time=3-0:00:00
cd ~/monte-carlo-techniques-2021/Project
/software/anaconda3/bin/python3 heatbath_finetuning.py -iw 3 -hw 17 -ki 0.08 -kf 0.18 -li 1 -lf 2 -la 3