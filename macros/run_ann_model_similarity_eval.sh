#!/bin/bash
#SBATCH --job-name resnet_CKA
#SBATCH --time 1-00:00:00
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 32
#SBATCH --hint=nomultithread
#SBATCH --output ann_model_similarity_eval.logs/model_similarity_%a.out
#SBATCH --account dbic
#SBATCH --gres gpu:1
#SBATCH --requeue
#SBATCH --partition a100_preemptable
#SBATCH --array 0-13

# This is a SLRUM batch job submission script for running on an HPC system. Other job submission
# systems are also popular, but they all function according to more or less the same principles
# and have equivalent configuration options you could substitute for the above.

hostname

# activate the conda environment with the geometry_vs_topography library
source /optnfs/common/miniconda3/etc/profile.d/conda.sh
conda activate vissl

echo "Start time:"
date

set -x

DATA_SRC=$(cat ../config.json | \
    python3 -c "import sys, json; print(json.load(sys.stdin)['hcp_participant_data']['S1200_imaging'])")

SCRATCH_DIR=$(cat ../config.json | \
    python3 -c "import sys, json; print(json.load(sys.stdin)['scratch_ssd'])")

# Garbage collection
cleanup() {
    rm -rf $SCRATCH_DIR
}
#trap cleanup EXIT

mkdir -p $SCRATCH_DIR

python -u ../scripts/eval_ann_model_similarities.py

echo "End time:"
date

