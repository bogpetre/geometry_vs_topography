#!/bin/bash
#SBATCH --job-name RSA
#SBATCH --time 2-00:00:00
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem=64G
#SBATCH --hint=nomultithread
#SBATCH --output hcp25_4.logs/bsc_%a.out
#SBATCH --account dbic
#SBATCH --array 1-208

# This is a SLRUM batch job submission script for running on an HPC system. Other job submission
# systems are also popular, but they all function according to more or less the same principles
# and have equivalent configuration options you could substitute for the above.
#
# This script assumes you've already run "first" (subject) level GLMs and corresponding RSA, e.g.
# by invoking the run_rsn_hcp_msmall_grayord_spm.sh macro

hostname

source /optnfs/common/miniconda3/etc/profile.d/conda.sh
conda activate env39

module load freesurfer/7.4.1
module load matlab

echo "Start time:"
date

set -x

d=25

DATA_SRC=$(cat ../config.json | \
    python3 -c "import sys, json; print(json.load(sys.stdin)['hcp_participant_data']['S1200_imaging'])")

OUT_DIR=../derivatives/restingstate2/hcp${d}/

ATLAS=$(cat ../config.json | \
    python3 -c "import sys, json; print(json.load(sys.stdin)['canlab2024']['path'])")

HCP_RESOURCES=$(cat ../config.json | \
    python3 -c "import sys, json; print(json.load(sys.stdin)['hcp_participant_data']['HCP_Resources'])")
RSN_TEMPLATE=$HCP_RESOURCES/GroupAvg/HCP_PTN1200/groupICA/groupICA_3T_HCP1200_MSMAll_d${d}.ica/melodic_IC.dscalar.nii

HCP_DIR=$(python3 -c "import hcp_utils; from importlib_resources import files; print(files('hcp_utils'))")
SURF_LEFT=$HCP_DIR/data/S1200.L.midthickness_MSMAll.32k_fs_LR.surf.gii
SURF_RIGHT=$HCP_DIR/data/S1200.R.midthickness_MSMAll.32k_fs_LR.surf.gii

sid_list1=($(cat ../resources/paired_sid.csv | awk -F, '{print $1}'))
sid_list2=($(cat ../resources/paired_sid.csv | awk -F, '{print $2}'))
SID1=${sid_list1[$[$SLURM_ARRAY_TASK_ID-1]]}
SID2=${sid_list2[$[$SLURM_ARRAY_TASK_ID-1]]}

space_avail=$(df -kT /scratch | tail -n 1 | awk '{print $5}')
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
trap cleanup EXIT
mkdir -p $SCRATCH_DIR

wb_command -cifti-all-labels-to-rois $ATLAS 1 $SCRATCH_DIR/parcels.dscalar.nii

mkdir -p $OUT_DIR/bsc/whitened_betas/cosine/
pairwise_parcel_op $OUT_DIR/results/$SID1/whitened_betas/cifti_math_results.dscalar.nii \
    $OUT_DIR/results/$SID2/whitened_betas/cifti_math_results.dscalar.nii \
    $OUT_DIR/bsc/whitened_betas/cosine/${SID1}_v_${SID2}.dscalar.nii \
    -s $SURF_LEFT $SURF_RIGHT \
    -p $ATLAS \
    -o cosine

wb_command -cifti-stats $OUT_DIR/bsc/whitened_betas/cosine/${SID1}_v_${SID2}.dscalar.nii \
    -reduce MEAN \
    -roi $SCRATCH_DIR/parcels.dscalar.nii \
    > $OUT_DIR/bsc/whitened_betas/cosine/${SID1}_v_${SID2}_cosim.tsv

../bin/rdm_similarity32 \
    $OUT_DIR/results/${SID1}/whitened_betas/crossnobis_distance.csv \
    $OUT_DIR/results/${SID2}/whitened_betas/crossnobis_distance.csv \
    $OUT_DIR/results/${SID1}/whitened_betas/whitening_matrix_out.bin \
    $OUT_DIR/results/${SID2}/whitened_betas/whitening_matrix_out.bin \
    $OUT_DIR/results/${SID1}/whitened_betas/whitening_matrix_out.json \
    $OUT_DIR/results/${SID2}/whitened_betas/whitening_matrix_out.json \
    $OUT_DIR/bsc/whitened_betas/cosine/${SID1}_v_${SID2}_wuc.tsv 0

mkdir -p $OUT_DIR/bsc/whitened_betas/correlation/
pairwise_parcel_op $OUT_DIR/results/$SID1/whitened_betas/cifti_math_results.dscalar.nii \
    $OUT_DIR/results/$SID2/whitened_betas/cifti_math_results.dscalar.nii \
    $OUT_DIR/bsc/whitened_betas/correlation/${SID1}_v_${SID2}.dscalar.nii \
    -s $SURF_LEFT $SURF_RIGHT \
    -p $ATLAS \
    -o correlation

wb_command -cifti-stats $OUT_DIR/bsc/whitened_betas/correlation/${SID1}_v_${SID2}.dscalar.nii \
    -reduce MEAN \
    -roi $SCRATCH_DIR/parcels.dscalar.nii \
    > $OUT_DIR/bsc/whitened_betas/correlation/${SID1}_v_${SID2}_r.tsv
    
mkdir -p $OUT_DIR/bsc/standardized_betas/cosine
pairwise_parcel_op $OUT_DIR/results/$SID1/standardized_betas/cifti_math_results.dscalar.nii \
    $OUT_DIR/results/$SID2/standardized_betas/cifti_math_results.dscalar.nii \
    $OUT_DIR/bsc/standardized_betas/cosine/${SID1}_v_${SID2}.dscalar.nii \
    -s $SURF_LEFT $SURF_RIGHT \
    -p $ATLAS \
    -o cosine

wb_command -cifti-stats $OUT_DIR/bsc/standardized_betas/cosine/${SID1}_v_${SID2}.dscalar.nii \
    -reduce MEAN \
    -roi $SCRATCH_DIR/parcels.dscalar.nii \
    > $OUT_DIR/bsc/standardized_betas/cosine/${SID1}_v_${SID2}_cosim.tsv

../bin/rdm_similarity32 \
    $OUT_DIR/results/${SID1}/standardized_betas/standardized_distance.csv \
    $OUT_DIR/results/${SID2}/standardized_betas/standardized_distance.csv \
    $OUT_DIR/results/${SID1}/standardized_betas/whitening_matrix_out.bin \
    $OUT_DIR/results/${SID2}/standardized_betas/whitening_matrix_out.bin \
    $OUT_DIR/results/${SID1}/standardized_betas/whitening_matrix_out.json \
    $OUT_DIR/results/${SID2}/standardized_betas/whitening_matrix_out.json \
    $OUT_DIR/bsc/standardized_betas/cosine/${SID1}_v_${SID2}_wuc.tsv 0

mkdir -p $OUT_DIR/bsc/standardized_betas/correlation
pairwise_parcel_op $OUT_DIR/results/$SID1/standardized_betas/cifti_math_results.dscalar.nii \
    $OUT_DIR/results/$SID2/standardized_betas/cifti_math_results.dscalar.nii \
    $OUT_DIR/bsc/standardized_betas/correlation/${SID1}_v_${SID2}.dscalar.nii \
    -s $SURF_LEFT $SURF_RIGHT \
    -p $ATLAS \
    -o correlation

wb_command -cifti-stats $OUT_DIR/bsc/standardized_betas/correlation/${SID1}_v_${SID2}.dscalar.nii \
    -reduce MEAN \
    -roi $SCRATCH_DIR/parcels.dscalar.nii \
    > $OUT_DIR/bsc/standardized_betas/correlation/${SID1}_v_${SID2}_r.tsv

#END


