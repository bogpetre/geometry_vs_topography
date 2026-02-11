#!/bin/bash
#SBATCH --job-name 7SPM_GLMs
#SBATCH --time 1-00:00:00
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem=48G
#SBATCH --hint=nomultithread
#SBATCH --output single_blk_grayord_spm10.logs/firstlvl_%a.out
#SBATCH --account dbic
#SBATCH --array 1-416

hostname

# load your conda environment with geometry_vs_topography library
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
    python3 -c "import sys, json; print(json.load(sys.stdin)['hcp_participant_data']['S1200_imaging'])")

OUT_DIR=../derivatives/single_blocks_msmall_grayord_spm/

ATLAS=$(cat ../config.json | \
    python3 -c "import sys, json; print(json.load(sys.stdin)['canlab2024']['path'])")

sid_list=($(cat ../resources/paired_sid.csv | awk -F, '{print $1"\n"$2}'))
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
cleanup() {
    rm -rf $SCRATCH_DIR
}
trap cleanup EXIT # comment to retain intermediate analysis files
#SCRATCH_DIR=$OUT_DIR/workdir/$SID # uncomment to retain intermediate analysis files
mkdir -p $SCRATCH_DIR


TASKS=('EMOTION' 'GAMBLING' 'SOCIAL' 'LANGUAGE' 'RELATIONAL' 'MOTOR' 'WM')
directions=('LR' 'RL')

if [ ! -e $OUT_DIR/results/${SID1}/all_tasks/whitened_contrasts/binary_clf_performance.csv ]; then
    python -u ../scripts/hcp_glm_msmall_grayord_spm_single_blocks.py --subject_ids ${SID1} --out $OUT_DIR \
        --scratch $SCRATCH_DIR --atlas $ATLAS --tasks ${TASKS[@]} --n_cpus 1 \
        --config ../config.json
fi

