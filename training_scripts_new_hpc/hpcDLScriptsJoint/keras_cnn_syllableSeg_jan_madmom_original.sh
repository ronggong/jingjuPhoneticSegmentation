#!/bin/bash

#SBATCH -J jjssvlw
#SBATCH -p high
#SBATCH -N 1
#SBATCH --workdir=/homedtic/rgong/cnnSyllableSeg/jingjuPhoneticSegmentation
#SBATCH --gres=gpu:1
#SBATCH --sockets-per-node=1
#SBATCH --cores-per-socket=1
#SBATCH --threads-per-core=1

# Output/Error Text
# ----------------
#SBATCH -o /homedtic/rgong/cnnSyllableSeg/out/jjssvlw.%N.%J.%u.out # STDOUT
#SBATCH -e /homedtic/rgong/cnnSyllableSeg/out/jjssvlw.%N.%J.%u.err # STDERR

# anaconda environment
export PATH=/homedtic/rgong/anaconda2/bin:$PATH
source activate /homedtic/rgong/keras_env

python /homedtic/rgong/cnnSyllableSeg/jingjuPhoneticSegmentation/training_scripts_new_hpc/hpcDLScriptsJoint/keras_cnn_syllableSeg_jan_madmom_original.py