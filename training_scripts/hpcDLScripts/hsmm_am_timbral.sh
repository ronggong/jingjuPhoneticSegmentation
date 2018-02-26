#!/bin/bash

export PATH=/homedtic/rgong/anaconda2/bin:$PATH
source activate keras_env

printf "Removing local scratch directories if exist...\n"
if [ -d /scratch/acousticModelsTraining ]; then
        rm -Rf /scratch/acousticModelsTraining
fi

# Second, replicate the structure of the experiment's folder:
# -----------------------------------------------------------
mkdir /scratch/acousticModelsTraining
mkdir /scratch/acousticModelsTraining/syllableSeg


printf "Copying feature files into scratch directory...\n"
# Third, copy the experiment's data:
# ----------------------------------
start=`date +%s`
cp -rp /homedtic/rgong/acousticModelsTraining/dataset/feature_hsmm_am.h5 /scratch/acousticModelsTraining/
end=`date +%s`

printf "Finish copying feature files into scratch directory...\n"
printf $((end-start))


#$ -N hsmm_am
#$ -q default.q
#$ -l h=node05

# Output/Error Text
# ----------------
#$ -o /homedtic/rgong/acousticModelsTraining/out/hsmm_am.$JOB_ID.out
#$ -e /homedtic/rgong/acousticModelsTraining/error/hsmm_am.$JOB_ID.err

python /homedtic/rgong/acousticModelsTraining/training_scripts/hpcDLScripts/hsmm_am_timbral.py

# Clean the crap:
# ---------------
printf "Removing local scratch directories...\n"
if [ -d /scratch/acousticModelsTraining ]; then
        rm -Rf /scratch/acousticModelsTraining
fi
printf "Job done. Ending at `date`\n"
