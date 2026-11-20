#!/bin/bash
#SBATCH --job-name ResNet18_ImageNet
#SBATCH --time 5:00:00
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 32
#SBATCH --requeue
#SBATCH --output ann.logs/ann_%a_linear_eval_cleanup.out
#SBATCH --account dbic
#SBATCH --partition preempt_wager
#SBATCH --gres gpu:1
#SBATCH --array 28-42
#SBATCH --dependency=6906617

hostname
date
set -x

source /optnfs/common/miniconda3/etc/profile.d/conda.sh
conda activate vissl

SCRATCH_DIR=$(cat ../config.json | \
    python3 -c "import sys, json; print(json.load(sys.stdin)['scratch_ssd'])")

if [ "x$SCRATCH_DIR" == "x" ]; then
    echo "scratch_ssd is undefined in config.json. Attempting to use default scratch. If this is on a network filesystem this may be extremely slow"
    SCRATCH_DIR=$(cat ../config.json | \ 
        python3 -c "import sys, json; print(json.load(sys.stdin)['scratch'])")
   if [ "x$SCRATCH_DIR" == "x" ]; then
        echo "scratch is also undefined in config.json. Please configure your scratch directory parameters correctly"
        exit
   fi
fi

export SCRATCH_DIR=$SCRATCH_DIR
if [ ! -e $SCRATCH_DIR/ILSVRC2012/validation/ ]; then
    mkdir -p $SCRATCH_DIR/ILSVRC2012/validation/

    imagenet_data_dir=$(cat ../config.json | \
        python3 -c "import sys, json; print(json.load(sys.stdin)['imagenet_path'])")

    rsync -av --progress $imagenet_data_dir/ILSVRC2012_img_val.tar $SCRATCH_DIR/

    tar xf $SCRATCH_DIR/ILSVRC2012_img_val.tar -C $SCRATCH_DIR/ILSVRC2012/validation/
    cd $SCRATCH_DIR/ILSVRC2012/validation/

    # creates category directories and moves jpgs into category directories
    $imagenet_data_dir/valprep.sh
    cd -
fi

ST_BASE_FS=$(cat ../config.json | \
    python3 -c "import sys, json; print(json.load(sys.stdin)['TDANN_repo_path'])")
export ST_BASE_FS

taskid=${SLURM_ARRAY_TASK_ID}

if [ $taskid -lt 14 ]; then
   spatial="spatial"
   seed=$taskid
   ckptdir=simclr_${spatial}_resnet18_swappedon_SineGrating2019_isoswap_3_seed_${seed}_linear_eval
   epochs=200
   weightsdir=$(echo $ckptdir | sed 's/_linear_eval//' | sed 's|linear_eval/||')
   echo "Training TDANN"
elif [ $taskid -lt 28 ]; then
   spatial="nonspatial"
   seed=$[$taskid-14]
   ckptdir=simclr_${spatial}_resnet18_swappedon_SineGrating2019_isoswap_3_seed_${seed}_linear_eval
   epochs=200
   weightsdir=$(echo $ckptdir | sed 's/_linear_eval//' | sed 's|linear_eval/||')
   echo "Training vanilla ResNet-18"
elif [ $taskid -lt 42 ]; then
   spatial="nonspatial"
   seed=$[$taskid-28]
   ckptdir=simclr_${spatial}_resnet18_swappedon_SineGrating2019_isoswap_3_seed_${seed}_60_epochs_linear_eval
   epochs=150
   weightsdir=$(echo $ckptdir | sed 's/_linear_eval//' | sed 's|linear_eval/||')
else
   echo "This script is only configured to run with task IDs less than 42. Please decrease the task iteration and rerun"
   exit
fi

ckptdir=$(readlink -f ../derivatives/models/checkpoints/linear_eval/$ckptdir)
weightsdir=$(readlink -f ../derivatives/models/checkpoints/$weightsdir)

if [ $taskid -gt 27 ] && [ $taskid -lt 42 ] && [ ! -e $weightsdir ]; then
    oldweights=$(echo $weightsdir | sed 's/_60_epochs//')
    mkdir -p $weightsdir
    ln -s $oldweights/model_phase60.torch $weightsdir/checkpoint.torch
fi

cd $ST_BASE_FS
time python -u train.py \
    config=linear_eval/rsa_vs_topo/simclr_spatial_resnet18_swappedon_SineGrating2019_isoswap_3_linear_eval.yaml \
    config.CHECKPOINT.DIR=$ckptdir \
    config.MODEL.WEIGHTS_INIT.PARAMS_FILE=$weightsdir \
    config.SEED_VALUE=${seed} \
    config.DATA.NUM_DATALOADER_WORKERS=${SLURM_CPUS_PER_TASK}

date
