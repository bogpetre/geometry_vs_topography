#!/bin/bash
#SBATCH --job-name RSN_RSA
#SBATCH --time 2-00:00:00
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --hint=nomultithread
#SBATCH --output rsn25_3f.logs/all_bsc_%a.out
#SBATCH --account dbic
#SBATCH --array 1-1114
#SBATCH --exclude=
#SBATCH --dependency=7195285

# This is a SLRUM batch job submission script for running on an HPC system. Other job submission
# systems are also popular, but they all function according to more or less the same principles
# and have equivalent configuration options you could substitute for the above.
#
# This script assumes you've already run "first" (subject) level GLMs and corresponding RSA, e.g.
# by invoking the run_rsn_hcp_msmall_grayord_spm.sh macro
#
# This script evaluates all pairwise combinations of participants for heritability analysis
# purposes

hostname

source /optnfs/common/miniconda3/etc/profile.d/conda.sh
conda activate env39

module load matlab
module load freesurfer/7.4.1

# This seems to take about 4 hours to run using 48G of memory and
# 1x 3GHz core of an AMD EPYC 7532 CPU (4800 bogomips), but YMMV.
echo "Start time:"
date

set -x

d=25

ROOT=/dartfs-hpc/rc/lab/C/CANlab/labdata/projects/bogdan_hcp_glm/

DATA_SRC=$(cat ../config.json | \
    python3 -c "import sys, json; print(json.load(sys.stdin)['hcp_participant_data']['S1200_imaging'])")

OUT_DIR=../derivatives/restingstate/hcp${d}/

ATLAS=$(cat ../config.json | \
    python3 -c "import sys, json; print(json.load(sys.stdin)['canlab2024']['path'])")

HCP_RESOURCES=$(cat ../config.json | \
    python3 -c "import sys, json; print(json.load(sys.stdin)['hcp_participant_data']['HCP_Resources'])")
RSN_TEMPLATE=$HCP_RESOURCES/GroupAvg/HCP_PTN1200/groupICA/groupICA_3T_HCP1200_MSMAll_d${d}.ica/melodic_IC.dscalar.nii

HCP_DIR=$(python3 -c "import hcp_utils; from importlib_resources import files; print(files('hcp_utils'))")
SURF_LEFT=$HCP_DIR/data/S1200.L.midthickness_MSMAll.32k_fs_LR.surf.gii
SURF_RIGHT=$HCP_DIR/data/S1200.R.midthickness_MSMAll.32k_fs_LR.surf.gii

sid_list1=($(ls $DATA_SRC))
iter=$[$SLURM_ARRAY_TASK_ID-1]
sid_list2=${sid_list1[@]:$[${iter}+1]:${#sid_list1[@]}}
SID1=${sid_list1[$[$SLURM_ARRAY_TASK_ID-1]]}

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


if [ ! -e $OUT_DIR/results/$SID1/whitened_betas/cifti_math_results.dscalar.nii ]; then
    exit
fi

# copy prime subjects data to scratch for faster access
mkdir -p $SCRATCH_DIR/whitened/ $SCRATCH_DIR/standard/
cp $OUT_DIR/results/${SID1}/whitened_betas/crossnobis_distance.csv \
    $OUT_DIR/results/${SID1}/whitened_betas/whitening_matrix_out.bin \
    $OUT_DIR/results/${SID1}/whitened_betas/whitening_matrix_out.json \
    $SCRATCH_DIR/whitened/
cp $OUT_DIR/results/${SID1}/standardized_betas/standardized_distance.csv \
   $OUT_DIR/results/${SID1}/standardized_betas/whitening_matrix_out.bin \
   $OUT_DIR/results/${SID1}/standardized_betas/whitening_matrix_out.json \
   $SCRATCH_DIR/standard/

# Topographic similarity is estimated using python scripts which are slow to spin up, so we do it on demand instead
# in matlab
mkdir -p $OUT_DIR/bsc_all/whitened_betas/cosine/
if [ -e $OUT_DIR/bsc_all/whitened_betas/cosine/${SID1}/ ]; then # take a count of how many files are within
    wcnt=$(ls $OUT_DIR/bsc_all/whitened_betas/cosine/${SID1}/ | wc -w);
else
    wcnt=0
fi
if [ -e $OUT_DIR/bsc_all/standardized_betas/cosine/${SID1}/ ]; then # take a count of how many files are within
    scnt=$(ls $OUT_DIR/bsc_all/standardized_betas/cosine/${SID1}/ | wc -w);
else
    scnt=0
fi

for SID2 in ${sid_list2[@]}; do 
    if [ ! -e $OUT_DIR/results/$SID2/whitened_betas/cifti_math_results.dscalar.nii ]; then
        continue
    fi
    if [ ! -e $OUT_DIR/bsc_all/whitened_betas/cosine/${SID1}/${SID1}_v_${SID2}_wuc.tsv ]; then
        mkdir -p $SCRATCH_DIR/whitened/$SID2
        cp $OUT_DIR/results/${SID2}/whitened_betas/crossnobis_distance.csv \
            $OUT_DIR/results/${SID2}/whitened_betas/whitening_matrix_out.bin \
            $OUT_DIR/results/${SID2}/whitened_betas/whitening_matrix_out.json \
            $SCRATCH_DIR/whitened/${SID2}/

        mkdir -p $OUT_DIR/bsc_all/whitened_betas/cosine/${SID1}/
        time ../bin/rdm_similarity64 \
            $SCRATCH_DIR/whitened/crossnobis_distance.csv \
            $SCRATCH_DIR/whitened/${SID2}/crossnobis_distance.csv \
            $SCRATCH_DIR/whitened/whitening_matrix_out.bin \
            $SCRATCH_DIR/whitened/${SID2}/whitening_matrix_out.bin \
            $SCRATCH_DIR/whitened/whitening_matrix_out.json \
            $SCRATCH_DIR/whitened/${SID2}/whitening_matrix_out.json \
            $OUT_DIR/bsc_all/whitened_betas/cosine/${SID1}/${SID1}_v_${SID2}_wuc.tsv 0 

        rm -rf $SCRATCH_DIR/whitened/${SID2}/ &
    fi
    if [ ! -e $OUT_DIR/bsc_all/standardized_betas/cosine/${SID1}/${SID1}_v_${SID2}_wuc.tsv ]; then
        mkdir -p $SCRATCH_DIR/standard/$SID2
        cp $OUT_DIR/results/${SID2}/standardized_betas/standardized_distance.csv \
           $OUT_DIR/results/${SID2}/standardized_betas/whitening_matrix_out.bin \
           $OUT_DIR/results/${SID2}/standardized_betas/whitening_matrix_out.json \
           $SCRATCH_DIR/standard/$SID2/

        mkdir -p $OUT_DIR/bsc_all/standardized_betas/cosine/${SID1}/
        time ../bin/rdm_similarity64 \
            $SCRATCH_DIR/standard/standardized_distance.csv \
            $SCRATCH_DIR/standard/${SID2}/standardized_distance.csv \
            $SCRATCH_DIR/standard/whitening_matrix_out.bin \
            $SCRATCH_DIR/standard/${SID2}/whitening_matrix_out.bin \
            $SCRATCH_DIR/standard/whitening_matrix_out.json \
            $SCRATCH_DIR/standard/${SID2}/whitening_matrix_out.json \
            $OUT_DIR/bsc_all/standardized_betas/cosine/${SID1}/${SID1}_v_${SID2}_wuc.tsv 0 
        rm -rf $SCRATCH_DIR/standard/${SID2}/ &
    fi
done

if [ $wcnt -lt $(ls $OUT_DIR/bsc_all/whitened_betas/cosine/${SID1}/ | wc -w) ]; then
    cat $OUT_DIR/bsc_all/whitened_betas/cosine/${SID1}/* > $OUT_DIR/bsc_all/whitened_betas/cosine/${SID1}_v_all.tsv
fi
if [ $scnt -lt $(ls $OUT_DIR/bsc_all/standardized_betas/cosine/${SID1}/ | wc -w) ]; then
    cat $OUT_DIR/bsc_all/standardized_betas/cosine/${SID1}/* > $OUT_DIR/bsc_all/standardized_betas/cosine/${SID1}_v_all.tsv
fi

# if we have nothing to compare, we make a placeholder file
if [ ! -e $OUT_DIR/bsc_all/standardized_betas/cosine/${SID1}_v_all.tsv ]; then
    touch $OUT_DIR/bsc_all/standardized_betas/cosine/${SID1}_v_all.tsv
fi
if [ ! -e $OUT_DIR/bsc_all/whitened_betas/cosine/${SID1}_v_all.tsv ]; then
    touch $OUT_DIR/bsc_all/whitened_betas/cosine/${SID1}_v_all.tsv
fi

echo "End time:"
date

