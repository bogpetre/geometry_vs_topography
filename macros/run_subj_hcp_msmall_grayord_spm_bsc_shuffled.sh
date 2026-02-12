#!/bin/bash
#SBATCH --job-name BSC
#SBATCH --time 1-00:00:00
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --hint=nomultithread
#SBATCH --output hcp_glm_msmall_grayord_spm.logs/bsc_null_%a.out
#SBATCH --account dbic
#SBATCH --array 1-100

# This is a SLRUM batch job submission script for running on an HPC system. Other job submission
# systems are also popular, but they all function according to more or less the same principles
# and have equivalent configuration options you could substitute for the above.
#
# This script assumes you've already run "first" (subject) level GLMs and corresponding RSA, e.g.
# by invoking the run_subj_hcp_msmall_grayord_spm.sh macro.
#
# Here we compute similarity measures among random sets of task conditions. This provides a null
# similarity baseline against which to compare observed similarity measures. We do not use this
# for serious inference (because our questions aren't about whether or not similarities are
# different from chance, but rather how do similarities vary w.r.t. one another or with
# gradients, so we mainly draw inference over sample and spatial variance, not task variance),
# but we do use these results for some basic sanity checks and to provide a reference scale for
# some of our similarity measures.

hostname

# activate the conda environment with the geometry_vs_topography library
source /optnfs/common/miniconda3/etc/profile.d/conda.sh
conda activate env39

echo "Start time:"
date

set -x

OUT_DIR=../derivatives/hcp_glm_msmall_grayord_spm/

TASK_ID=$[$SLURM_ARRAY_TASK_ID-1]

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

seed=$TASK_ID
rand_ind=$(
    seq 0 22 | shuf --random-source=<(openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt </dev/zero 2>/dev/null)
  )

shuffle_str=$(echo ${rand_ind[@]} | sed 's/\ /,/g')

mkdir -p $OUT_DIR/bsc_null/seed${seed}/
echo $shuffle_str > $OUT_DIR/bsc_null/seed${seed}/shuffle_ind.csv
echo $OUT_DIR/bsc_null/seed${seed}/shuffle_ind.csv


# Below we compute WUC of whitened or standardized RDMs using shuffled indices.
# We do compute cosine similarities of spatial maps because we need to handle
# balancing and permutation of labels more carefully than is possible using our
# CLI utilities. Instead we save the permutation order and will handle cosine
# similarity computations directly in matlab

sid_list1=($(cat ../resources/paired_sid.csv | awk -F, '{print $1}'))
sid_list2=($(cat ../resources/paired_sid.csv | awk -F, '{print $2}'))

for i in $(seq 0 206); do

    SID1=${sid_list1[$i]}
    SID2=${sid_list2[$i]}

    if [ ! -e $OUT_DIR/results/$SID1/all_tasks/whitened_contrasts/merged_cifti.dscalar.nii ]; then
        exit
    elif [ ! -e $OUT_DIR/results/$SID2/all_tasks/whitened_contrasts/merged_cifti.dscalar.nii ]; then
        exit
    fi

    # whitened betas
    mkdir -p $OUT_DIR/bsc_null/seed${seed}/whitened_betas/cosine/

    cp $OUT_DIR/results/${SID1}/all_tasks/rsa/crossnobis/whitening_matrix_out.bin $SCRATCH_DIR/${SID1}_wh_whitening_matrix_out.bin
    cp $OUT_DIR/results/${SID2}/all_tasks/rsa/crossnobis/whitening_matrix_out.bin $SCRATCH_DIR/${SID2}_wh_whitening_matrix_out.bin
    ../bin/rdm_similarity64 \
        $OUT_DIR/results/${SID1}/all_tasks/rsa/crossnobis/crossnobis_distance.csv \
        $OUT_DIR/results/${SID2}/all_tasks/rsa/crossnobis/crossnobis_distance.csv \
        $SCRATCH_DIR/${SID1}_wh_whitening_matrix_out.bin \
        $SCRATCH_DIR/${SID2}_wh_whitening_matrix_out.bin \
        $OUT_DIR/results/${SID1}/all_tasks/rsa/crossnobis/whitening_matrix_out.json \
        $OUT_DIR/results/${SID2}/all_tasks/rsa/crossnobis/whitening_matrix_out.json \
        $OUT_DIR/bsc_null/seed${seed}/whitened_betas/cosine/${SID1}_v_${SID2}_wuc.tsv 0 \
        1,1,2,2,3,3,4,4,5,5,6,6,6,6,6,7,7,7,7,7,7,7,7 1 $shuffle_str

    # standardized betas
    mkdir -p $OUT_DIR/bsc_null/seed${seed}/standardized_betas/cosine/

    cp $OUT_DIR/results/${SID1}/all_tasks/rsa/stddist/whitening_matrix_out.bin $SCRATCH_DIR/${SID1}_std_whitening_matrix_out.bin
    cp $OUT_DIR/results/${SID2}/all_tasks/rsa/stddist/whitening_matrix_out.bin $SCRATCH_DIR/${SID2}_std_whitening_matrix_out.bin 
    ../bin/rdm_similarity64 \
        $OUT_DIR/results/${SID1}/all_tasks/rsa/stddist/standardized_distance.csv \
        $OUT_DIR/results/${SID2}/all_tasks/rsa/stddist/standardized_distance.csv \
        $SCRATCH_DIR/${SID1}_std_whitening_matrix_out.bin \
        $SCRATCH_DIR/${SID2}_std_whitening_matrix_out.bin \
        $OUT_DIR/results/${SID1}/all_tasks/rsa/stddist/whitening_matrix_out.json \
        $OUT_DIR/results/${SID2}/all_tasks/rsa/stddist/whitening_matrix_out.json \
        $OUT_DIR/bsc_null/seed${seed}/standardized_betas/cosine/${SID1}_v_${SID2}_wuc.tsv 0 \
        1,1,2,2,3,3,4,4,5,5,6,6,6,6,6,7,7,7,7,7,7,7,7 1 $shuffle_str

done
echo "End time:"
date

