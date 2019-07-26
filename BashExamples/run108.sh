#!/bin/bash
#SBATCH -n 2 
#SBATCH -p standard
#SBATCH --mem=10gb
#SBATCH --time=3-00:00:00
#SBATCH -o tmp108.log
date
module load camb/1.0.6 openmpi/2.1.1/b1
cd /home/iflood/NLC/NLC_Ina/py/
mpirun -n 2 python MC_NLC_P1D_3param_pool.py --out_dir "run108" --z 2.4  --params 0 0 1 0 1 1 --err 0.5 --pos_method 2 --ndim 3 --nwalkers 10 --nsteps 400 --multiT
date
