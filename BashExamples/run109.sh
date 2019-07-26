#!/bin/bash
#SBATCH -p standard
#SBATCH --mem=10gb
#SBATCH --time=4:00:00
#SBATCH -o tmp109.log
date
hostname
module list
module load camb/1.0.6
cd /home/iflood/NLC/NLC_Ina/py/
python MC_NLC_P1D_post_3param.py --out_dir "run109" --in_dir "run108" --nsteps 1000 --CTSwitch
date
