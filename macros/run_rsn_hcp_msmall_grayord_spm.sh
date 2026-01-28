#!/bin/bash
#SBATCH --job-name RSN_RSA
#SBATCH --time 2-00:00:00
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem=64G
#SBATCH --hint=nomultithread
#SBATCH --output rsn25.logs/rsn25_%a.out
#SBATCH --account dbic
#SBATCH --array 2

# This is a SLRUM batch job submission script for running on an HPC system. Other job submission
# systems are also popular, but they all function according to more or less the same principles
# and have equivalent configuration options you could substitute for the above.

hostname

# activate the conda environment with the geometry_vs_topography library
source /optnfs/common/miniconda3/etc/profile.d/conda.sh
conda activate env39

# Make sure matlab and freesurfer are available on the system path. 
# This makes them available on the Dartmouth HPC
module load freesurfer/7.4.1
module load matlab

echo "Start time:"
date

set -x

d=25 # number of ICA components to use. Options: 15, 25, 50, 100, 150, 200, 300. Must match ICA templates available from PTN1200 release

DATA_SRC=$(cat ../config.json | \
    python3 -c "import sys, json; print(json.load(sys.stdin)['hcp_participant_data']['S1200_imaging'])")

OUT_DIR=../derivatives/restingstate/hcp${d}/

ATLAS=$(cat ../config.json | \
    python3 -c "import sys, json; print(json.load(sys.stdin)['canlab2024']['path'])")

HCP_RESOURCES=$(cat ../config.json | \
    python3 -c "import sys, json; print(json.load(sys.stdin)['hcp_participant_data']['HCP_Resources'])")
RSN_TEMPLATE=$HCP_RESOURCES/GroupAvg/HCP_PTN1200/groupICA/groupICA_3T_HCP1200_MSMAll_d${d}.ica/melodic_IC.dscalar.nii

sid_list=($(ls $DATA_SRC))
SID1=${sid_list[$[$SLURM_ARRAY_TASK_ID-1]]}

# The $TMPDIR env variable points to local scratch space, which is sometimes full.
# In the event there's not enough space we fall back to NFS scratch space. Substitute
# with your own equivalents.
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
#trap cleanup EXIT # comment to retain analysis intermediaries
SCRATCH_DIR=$OUT_DIR/workdir/$SID # uncomment to retain analysis intermediaries
mkdir -p $SCRATCH_DIR

if [ ! -e $OUT_DIR/results/$SID1/cifti_average_parcellated.txt ]; then
    python -u ../scripts/hcp_dual_regression_msmall_grayord_spm.py \
        --subject_ids ${SID1} --out $OUT_DIR \
        --scratch $SCRATCH_DIR/${SID1} --n_cpus 1 \
        --atlas $ATLAS --rsn_template $RSN_TEMPLATE \
        --config ../config.json
fi


