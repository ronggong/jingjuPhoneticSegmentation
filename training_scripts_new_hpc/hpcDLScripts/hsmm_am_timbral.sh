#!/bin/bash

#SBATCH -J AM
#SBATCH -p high
#SBATCH -N 1
#SBATCH --workdir=/homedtic/rgong/acousticModelsTraining
#SBATCH --gres=gpu:maxwell:1
#SBATCH --sockets-per-node=1
#SBATCH --cores-per-socket=2
#SBATCH --threads-per-core=2

# Output/Error Text
# ----------------
#SBATCH -o /homedtic/rgong/acousticModelsTraining/out/am.%N.%J.%u.out # STDOUT
#SBATCH -e /homedtic/rgong/acousticModelsTraining/out/am.%N.%J.%u.err # STDERR

# anaconda environment
export PATH=/homedtic/rgong/anaconda2/bin:$PATH
source activate /homedtic/rgong/keras_env

python /homedtic/rgong/acousticModelsTraining/training_scripts_new_hpc/hpcDLScripts/hsmm_am_timbral.py

