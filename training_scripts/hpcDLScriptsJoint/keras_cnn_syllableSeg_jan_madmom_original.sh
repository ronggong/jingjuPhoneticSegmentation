#!/bin/bash

export PATH=/homedtic/rgong/anaconda2/bin:$PATH
source activate keras_env

printf "Removing local scratch directories if exist...\n"
if [ -d /scratch/rgongcnnSyllableSeg_jan_joint ]; then
        rm -Rf /scratch/rgongcnnSyllableSeg_jan_joint
fi

# Second, replicate the structure of the experiment's folder:
# -----------------------------------------------------------
mkdir /scratch/rgongcnnSyllableSeg_jan_joint
mkdir /scratch/rgongcnnSyllableSeg_jan_joint/syllableSeg


printf "Copying feature files into scratch directory...\n"
# Third, copy the experiment's data:
# ----------------------------------
start=`date +%s`
cp -rp /homedtic/rgong/cnnSyllableSeg/syllableSeg/feature_all_joint.h5 /scratch/rgongcnnSyllableSeg_jan_joint/syllableSeg/
end=`date +%s`

printf "Finish copying feature files into scratch directory...\n"
printf $((end-start))


#$ -N jan_joint
#$ -q default.q
#$ -l debian8
#$ -l h=node11

# Output/Error Text
# ----------------
#$ -o /homedtic/rgong/cnnSyllableSeg/out/jan_joint.$JOB_ID.out
#$ -e /homedtic/rgong/cnnSyllableSeg/error/jan_joint.$JOB_ID.err

python /homedtic/rgong/cnnSyllableSeg/jingjuSyllabicSegmentation/training_scripts/hpcDLScriptsJoint/keras_cnn_syllableSeg_jan_madmom_original.py

# Clean the crap:
# ---------------
printf "Removing local scratch directories...\n"
if [ -d /scratch/rgongcnnSyllableSeg_jan_joint ]; then
        rm -Rf /scratch/rgongcnnSyllableSeg_jan_joint
fi
printf "Job done. Ending at `date`\n"
