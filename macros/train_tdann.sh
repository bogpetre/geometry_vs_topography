#!/bin/bash
#SBATCH --job-name ResNet18_ImageNet
#SBATCH --time 3-00:00:00
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 32
#SBATCH --output ann.logs/ann_%a.out
#SBATCH --account dbic
#SBATCH --partition a100_preemtable
#SBATCH --requeue
#SBATCH --gres gpu:1
#SBATCH --array 0-27

# This script trains TDANNs (if taskid < 14) and vanilla ResNets (otherwise) using the TDANN
# library, vissl (for self supervision) and hydra (for config file parsing). Refer to 
# the TDANN repository for instructions on setup and use. https://github.com/neuroailab/TDANN
# Note, that you will need to download the TDANN OSF data (for unit positions) and ImageNet
# training data for this script to run successfully. Make sure these have correct paths specified
# in your config.json file
#
# Note, this will copy imagenet data to scratch space. Best to spin up one of these jobs per
# compute node first, and wait for that operation to complete before spinning up more on the
# same compute node to avoid having multiple scripts simultaneously copying to scratch. Data is
# not deleted from scratch upon completion, nor copied over a second time if it is already
# there, so you can safely run multiple instances of this script concurrently once the imagenet
# data is copied over to scratch space.

hostname
date
set -x

source /optnfs/common/miniconda3/etc/profile.d/conda.sh
conda activate vissl

SCRATCH_DIR=$(cat ../config.json | \
    python3 -c "import sys, json; print(json.load(sys.stdin)['scratch_ssd'])

if [ "x$SCRATCH_DIR" == "x" ]; then
    echo "scratch_ssd is undefined in config.json. Attempting to use default scratch. If this is on a network filesystem this may be extremely slow"
    SCRATCH_DIR=$(cat ../config.json | \ 
        python3 -c "import sys, json; print(json.load(sys.stdin)['scratch'])
   if [ "x$SCRATCH_DIR" == "x" ]; then
        echo "scratch is also undefined in config.json. Please configure your scratch directory parameters correctly"
        exit
   fi
fi

if [ ! -e $SCRATCH_DIR/ILSVRC2012/train/ ]; then
    ../scripts/copy_imgnet_train_to_scratch.sh
fi

ST_BASE_FS=$(cat ../config.json | \
    python3 -c "import sys, json; print(json.load(sys.stdin)['TDANN_repo_path'])")
export ST_BASE_FS

taskid=${SLURM_ARRAY_TASK_ID}

if [ $taskid -lt 14 ]; then
   spatial="spatial"
   seed=$taskid
   echo "Training TDANN"
elif [ $taskid -lt 28 ]; then
   spatial="nonspatial"
   seed=$[$taskid-14]
   echo "Training vanilla ResNet-18"
else
   echo "This script is only configured to run with task IDs less than 27. Please decrease the task iteration and rerun"
   exit
fi

cd $ST_BASE_FS
mkdir -p $ST_BASE_FS/configs/config/rsa_vs_topo
cp -vn ../resources/simclr_${spatial}_resnet18_swappedon_SineGrating2019_isoswap_3.yaml $ST_BASE_FS/configs/config/rsa_vs_topo
time python -u train.py \
    config=rsa_vs_topo/simclr_${spatial}_resnet18_swappedon_SineGrating2019_isoswap_3 \
    config.CHECKPOINT.DIR=$(readlink -f ../derivatives/models/checkpoints/simclr_${spatial}_resnet18_swappedon_SineGrating2019_isoswap_3_seed_${seed}) \
    config.SEED_VALUE=${seed}

date
