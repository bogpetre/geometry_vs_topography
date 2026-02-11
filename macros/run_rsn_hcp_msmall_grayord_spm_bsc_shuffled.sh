#!/bin/bash
#SBATCH --job-name nullRSA
#SBATCH --time 2-00:00:00
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem=48G
#SBATCH --hint=nomultithread
#SBATCH --output rsn25_3.logs/bsc_null_%a.out
#SBATCH --account dbic
#SBATCH --array 1-100

# This is a SLRUM batch job submission script for running on an HPC system. Other job submission
# systems are also popular, but they all function according to more or less the same principles
# and have equivalent configuration options you could substitute for the above.
#
# This script assumes you've already run "first" (subject) level GLMs and corresponding RSA, e.g.
# by invoking the run_rsn_hcp_msmall_grayord_spm.sh macro
#
# Here we compute similarity measures among random sets of networks. This provides a null
# similarity baseline against which to compare observed similarity measures. We do not use this
# for serious inference (because our questions aren't about whether or not similarities are
# different from chance, but rather how do similarities vary w.r.t. one another or with
# gradients, so we mainly draw inference over sample and spatial variance, not network variance),
# but we do use these results for some basic sanity checks and to provide a reference scale for
# some of our similarity measures.


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

OUT_DIR=../derivatives/restingstate/hcp${d}/

TASK_ID=$[$SLURM_ARRAY_TASK_ID-1]

ATLAS=$(cat ../config.json | \
    python3 -c "import sys, json; print(json.load(sys.stdin)['canlab2024']['path'])")

HCP_RESOURCES=$(cat ../config.json | \
    python3 -c "import sys, json; print(json.load(sys.stdin)['hcp_participant_data']['HCP_Resources'])")
RSN_TEMPLATE=$HCP_RESOURCES/GroupAvg/HCP_PTN1200/groupICA/groupICA_3T_HCP1200_MSMAll_d${d}.ica/melodic_IC.dscalar.nii

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
trap cleanup EXIT
mkdir -p $SCRATCH_DIR

mkdir -p $OUT_DIR/bsc_null/seed${seed}/whitened_betas/cosine/
seed=$TASK_ID
rand_ind=$(
    seq 0 24 | shuf --random-source=<(openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt </dev/zero 2>/dev/null)
  )

shuffle_str=$(echo ${rand_ind[@]} | sed 's/\ /,/g')

mkdir -p $OUT_DIR/bsc_null/seed${seed}/
echo $shuffle_str > $OUT_DIR/bsc_null/seed${seed}/shuffle_ind.csv
echo $OUT_DIR/bsc_null/seed${seed}/shuffle_ind.csv


# Below we compute WUC of whitened or standardized RDMs using shuffled indices.
# We do compute cosine similarities of spatial maps for consistency with task
# analysis, where we can't handle task balancing well using CLI utilities. We do
# cosine similarities in matlab for resting state networks as well as a result.

sid_list1=($(cat ../resources/paired_sid.csv | awk -F, '{print $1}'))
sid_list2=($(cat ../resources/paired_sid.csv | awk -F, '{print $2}'))

for i in $(seq 0 206); do

    SID1=${sid_list1[$i]}
    SID2=${sid_list2[$i]}

    mkdir -p $OUT_DIR/bsc_null/seed${seed}/whitened_betas/cosine

    ../bin/rdm_similarity64 \
        $OUT_DIR/results/${SID1}/whitened_betas/crossnobis_distance.csv \
        $OUT_DIR/results/${SID2}/whitened_betas/crossnobis_distance.csv \
        $OUT_DIR/results/${SID1}/whitened_betas/whitening_matrix_out.bin \
        $OUT_DIR/results/${SID2}/whitened_betas/whitening_matrix_out.bin \
        $OUT_DIR/results/${SID1}/whitened_betas/whitening_matrix_out.json \
        $OUT_DIR/results/${SID2}/whitened_betas/whitening_matrix_out.json \
        $OUT_DIR/bsc_null/seed${seed}/whitened_betas/cosine/${SID1}_v_${SID2}_wuc.tsv 0 \
        - 1 $shuffle_str

    mkdir -p $OUT_DIR/bsc_null/seed${seed}/standardized_betas/cosine

    ../bin/rdm_similarity64 \
        $OUT_DIR/results/${SID1}/standardized_betas/standardized_distance.csv \
        $OUT_DIR/results/${SID2}/standardized_betas/standardized_distance.csv \
        $OUT_DIR/results/${SID1}/standardized_betas/whitening_matrix_out.bin \
        $OUT_DIR/results/${SID2}/standardized_betas/whitening_matrix_out.bin \
        $OUT_DIR/results/${SID1}/standardized_betas/whitening_matrix_out.json \
        $OUT_DIR/results/${SID2}/standardized_betas/whitening_matrix_out.json \
        $OUT_DIR/bsc_null/seed${seed}/standardized_betas/cosine/${SID1}_v_${SID2}_wuc.tsv 0 \
        - 1 $shuffle_str

done
