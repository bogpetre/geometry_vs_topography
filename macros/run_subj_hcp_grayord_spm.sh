#!/bin/bash
#SBATCH --job-name 7SPM_GLMs
#SBATCH --time 1-00:00:00
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem=48G
#SBATCH --hint=nomultithread
#SBATCH --output hcp_glm_grayord_spm_b.logs/firstlvl_bsc_%a.out
#SBATCH --account dbic
#SBATCH --array 1-208

# This is a SLRUM batch job submission script for running on an HPC system. Other job submission
# systems are also popular, but they all function according to more or less the same principles
# and have equivalent configuration options you could substitute for the above.

# This script runs first level GLMs, RSA and computes topographic similarities using anatomically 
# aligned (not-MSMAll) data. This is a sanity check that multimodal alignment doesn't drive our
# results

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
    python3 -c "import sys, json; print(json.load(sys.stdin)['hcp_participant_data']['S1200_imaging'])")

# this is where your files get saved
OUT_DIR=../derivatives/hcp_glm_grayord_spm/

ATLAS=$(cat ../config.json | \
    python3 -c "import sys, json; print(json.load(sys.stdin)['canlab2024']['path'])")

HCP_DIR=$(python3 -c "import hcp_utils; from importlib_resources import files; print(files('hcp_utils'))")
SURF_LEFT=$HCP_DIR/data/S1200.L.midthickness_MSMAll.32k_fs_LR.surf.gii
SURF_RIGHT=$HCP_DIR/data/S1200.R.midthickness_MSMAll.32k_fs_LR.surf.gii

sid_list1=($(cat ../resources/paired_sid.csv | awk -F, '{print $1}'))
sid_list2=($(cat ../resources/paired_sid.csv | awk -F, '{print $2}'))
SID1=${sid_list1[$[$SLURM_ARRAY_TASK_ID-1]]}
SID2=${sid_list2[$[$SLURM_ARRAY_TASK_ID-1]]}

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
        --scratch $SCRATCH_DIR --atlas $ATLAS --tasks ${TASKS[@]} --n_cpus 1 \
        --config ../config.json --anatomical
fi

if [ ! -e $OUT_DIR/results/$SID2/all_tasks/whitened_contrasts/merged_cifti.dscalar.nii ]; then
    python -u ../scripts/hcp_glm_msmall_grayord_spm.py --subject_ids ${SID2} --out $OUT_DIR \
        --scratch $SCRATCH_DIR --atlas $ATLAS --tasks ${TASKS[@]} --n_cpus 1 \
        --config ../config.json --anatomical
fi

if [ ! -e $OUT_DIR/results/$SID1/all_tasks/whitened_contrasts/merged_cifti.dscalar.nii ]; then
    exit
elif [ ! -e $OUT_DIR/results/$SID2/all_tasks/whitened_contrasts/merged_cifti.dscalar.nii ]; then
    exit
fi

# Below we compute correlation distance and cosine similarity of spatially whitened, standardized
# and raw betas. t-stats are standardized beta up to a scaling factor that reflects 
# the covariance structure of the relevant design columns. cosine similarity and correlation
# distance are both scale invariant, so computing cosine sim and correlation distance on
# t-stats is equivalent to doing so on standardized betas.

wb_command -cifti-all-labels-to-rois $ATLAS 1 $SCRATCH_DIR/parcels.dscalar.nii

# whitened betas
mkdir -p $OUT_DIR/bsc/whitened_betas/cosine/
pairwise_parcel_op $OUT_DIR/results/$SID1/all_tasks/whitened_contrasts/merged_cifti.dscalar.nii \
    $OUT_DIR/results/$SID2/all_tasks/whitened_contrasts/merged_cifti.dscalar.nii \
    $SCRATCH_DIR/${SID1}_v_${SID2}_wh.dscalar.nii \
    -s $SURF_LEFT $SURF_RIGHT \
    -p $ATLAS \
    -o cosine

wb_command -cifti-stats $SCRATCH_DIR/${SID1}_v_${SID2}_wh.dscalar.nii \
    -reduce MEAN \
    -roi $SCRATCH_DIR/parcels.dscalar.nii \
    > $OUT_DIR/bsc/whitened_betas/cosine/${SID1}_v_${SID2}_cosim.tsv

cp $OUT_DIR/results/${SID1}/all_tasks/rsa/crossnobis/whitening_matrix_out.bin $SCRATCH_DIR/${SID1}_wh_whitening_matrix_out.bin
cp $OUT_DIR/results/${SID2}/all_tasks/rsa/crossnobis/whitening_matrix_out.bin $SCRATCH_DIR/${SID2}_wh_whitening_matrix_out.bin
../bin/rdm_similarity64 \
    $OUT_DIR/results/${SID1}/all_tasks/rsa/crossnobis/crossnobis_distance.csv \
    $OUT_DIR/results/${SID2}/all_tasks/rsa/crossnobis/crossnobis_distance.csv \
    $SCRATCH_DIR/${SID1}_wh_whitening_matrix_out.bin \
    $SCRATCH_DIR/${SID2}_wh_whitening_matrix_out.bin \
    $OUT_DIR/results/${SID1}/all_tasks/rsa/crossnobis/whitening_matrix_out.json \
    $OUT_DIR/results/${SID2}/all_tasks/rsa/crossnobis/whitening_matrix_out.json \
    $OUT_DIR/bsc/whitened_betas/cosine/${SID1}_v_${SID2}_wuc.tsv 0 \
    1,1,2,2,3,3,4,4,5,5,6,6,6,6,6,7,7,7,7,7,7,7,7

# standardized betas
mkdir -p $OUT_DIR/bsc/standardized_betas/cosine
pairwise_parcel_op $OUT_DIR/results/$SID1/all_tasks/standardized_contrasts/merged_cifti.dscalar.nii \
    $OUT_DIR/results/$SID2/all_tasks/standardized_contrasts/merged_cifti.dscalar.nii \
    $SCRATCH_DIR/${SID1}_v_${SID2}_std.dscalar.nii \
    -s $SURF_LEFT $SURF_RIGHT \
    -p $ATLAS \
    -o cosine

wb_command -cifti-stats $SCRATCH_DIR/${SID1}_v_${SID2}_std.dscalar.nii \
    -reduce MEAN \
    -roi $SCRATCH_DIR/parcels.dscalar.nii \
    > $OUT_DIR/bsc/standardized_betas/cosine/${SID1}_v_${SID2}_cosim.tsv

cp $OUT_DIR/results/${SID1}/all_tasks/rsa/stddist/whitening_matrix_out.bin $SCRATCH_DIR/${SID1}_std_whitening_matrix_out.bin
cp $OUT_DIR/results/${SID2}/all_tasks/rsa/stddist/whitening_matrix_out.bin $SCRATCH_DIR/${SID2}_std_whitening_matrix_out.bin 
../bin/rdm_similarity64 \
    $OUT_DIR/results/${SID1}/all_tasks/rsa/stddist/standardized_distance.csv \
    $OUT_DIR/results/${SID2}/all_tasks/rsa/stddist/standardized_distance.csv \
    $SCRATCH_DIR/${SID1}_std_whitening_matrix_out.bin \
    $SCRATCH_DIR/${SID2}_std_whitening_matrix_out.bin \
    $OUT_DIR/results/${SID1}/all_tasks/rsa/stddist/whitening_matrix_out.json \
    $OUT_DIR/results/${SID2}/all_tasks/rsa/stddist/whitening_matrix_out.json \
    $OUT_DIR/bsc/standardized_betas/cosine/${SID1}_v_${SID2}_wuc.tsv 0 \
    1,1,2,2,3,3,4,4,5,5,6,6,6,6,6,7,7,7,7,7,7,7,7

echo "End time:"
date

