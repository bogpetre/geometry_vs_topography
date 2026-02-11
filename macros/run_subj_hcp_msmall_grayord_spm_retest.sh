#!/bin/bash
#SBATCH --job-name 7SPM_GLMs
#SBATCH --time 1-00:00:00
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem=48G
#SBATCH --hint=nomultithread
#SBATCH --output hcp_glm_msmall_grayord_spm_retest.logs/firstlvl_%a.out
#SBATCH --account dbic
#SBATCH --array 1-24
#SBATCH --dependency=7194045

# This is a SLRUM batch job submission script for running on an HPC system. Other job submission
# systems are also popular, but they all function according to more or less the same principles
# and have equivalent configuration options you could substitute for the above.
#
# This script runs first level GLMs using retest data from participants that returned to repeat
# the full HCP protocol. Runs very similarly to run_subj_hcp_msmall_grayord_spm.sh except with
# a different DATA_SRC path. We also incorporate some parallelization since we have fewer of 
# these jobs and can use more CPUs/job before saturating our allocation, but modify to taste.
# Change the n_cpus argument to hcp_glm_msmall_grayord_spm.py below accordingly.

hostname

# activate the conda environment with the geometry_vs_topography library
source /optnfs/common/miniconda3/etc/profile.d/conda.sh
conda activate env39

# Make sure matlab is available. This makes it available on the Dartmouth HPC
module load matlab

# This seems to take about 4 hours to run using 48G of memory and
# 1x 3GHz core of an AMD EPYC 7532 CPU (4800 bogomips), but YMMV.
echo "Start time:"
date

set -x

DATA_SRC=$(cat ../config.json | \
    python3 -c "import sys, json; print(json.load(sys.stdin)['hcp_participant_data']['retest_imaging'])")

# this is where your files get saved
OUT_DIR=../derivatives/retest/hcp_glm_msmall_grayord_spm/

ATLAS=$(cat ../config.json | \
    python3 -c "import sys, json; print(json.load(sys.stdin)['canlab2024']['path'])")

sid_list=($(../resources/paired_retest_iid_sid.csv | awk -F, '{print $1"\n"$2}'))
SID1=${sid_list[$[$SLURM_ARRAY_TASK_ID-1]]}

# The $TMPDIR env variable points to local scratch space, which is sometimes full.
# In the event there's not enough space we fall back to NFS scratch space. Substitute
# with your own equivalents.
space_avail=$(df -kT $TMPDIR | tail -n 1 | awk '{print $5}')
disk_space_req=400 # scratch space required in gigabytes, each run takes 22, and we might run 20 on a node
if [[ $space_avail -gt $(echo 1024*1024*$disk_space_req | bc -l) ]]; then
        # faster but more resource constrained
	SCRATCH_DIR=$TMPDIR/$(uuidgen)
else
        # slower, but much less of a problem running out of space
        SCRATCH_DIR=/dartfs-hpc/scratch/$(whoami)/$(uuidgen)
fi

# Garbage collection
cleanup() {
    rm -rf $SCRATCH_DIR
}
trap cleanup EXIT

mkdir -p $SCRATCH_DIR

TASKS=('EMOTION' 'GAMBLING' 'SOCIAL' 'LANGUAGE' 'RELATIONAL' 'MOTOR' 'WM')
directions=('LR' 'RL')

if [ ! -e $OUT_DIR/results/$SID1/all_tasks/whitened_contrasts/merged_cifti.dscalar.nii ]; then
    python -u ../scripts/hcp_glm_msmall_grayord_spm.py --subject_ids ${SID1} --out $OUT_DIR \
        --scratch $SCRATCH_DIR --atlas $ATLAS --tasks ${TASKS[@]} --n_cpus 4 \
        --data $DATA_SRC --config ../config.json
fi

echo "End time:"
date

