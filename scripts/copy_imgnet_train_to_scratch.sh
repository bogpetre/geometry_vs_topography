#!/bin/bash
# Bogdan Petre (Dec 10, 2025)
#
# Note: there is no cleanup script here, so this will persist in scratch
# if you don't do something about it. A simple solution is to call this script from a job
# script that includes this in its header (without comments):
#
# cleanup() {
#    rm -rf /scratch/$(whoami)/ILSVRC2012
# }
# trap cleanup EXIT
#
# This will remove the ILSVRC2012 directory when your SLURM job ends, regardless of whether
# it completes successfully or not. If you're running concurrent jobs on the same node
# though you'll need a different approach, because the earliest script to finish will
# delete data that's still needed by the laggards.


# gets parent directory of this script
dir=$(cd $(dirname $0) && pwd)

if [ "x$SCRATCH_DIR" == "x" ]; then
    echo "scratch_ssd is undefined in config.json. Attempting to use default scratch. If this is on a network filesystem this may be extremely slow"
    SCRATCH_DIR=$(cat ../config.json | \ 
        python3 -c "import sys, json; print(json.load(sys.stdin)['scratch'])
   if [ "x$SCRATCH_DIR" == "x" ]; then
        echo "scratch is also undefined in config.json. Please configure your scratch directory parameters correctly"
        exit
   fi
fi

imagenet_data_dir=$(cat $dir/../config.json | \
    python3 -c "import sys, json; print(json.load(sys.stdin)['imagenet_path'])


# copy tarball to scratch. Assume it lives in the same directory as this script
mkdir -p $SCRATCH_DIR/ILSVRC2012/train/
rsync -av --progress $imagenet_data_dir/ILSVRC2012_img_train.tar $SCRATCH_DIR/ILSVRC2012/

# extract tarball and pipe outputs (more tarballs) to another tar invocation
# to extract them to synset folders.
tar xf $SCRATCH_DIR/ILSVRC2012/ILSVRC2012_img_train.tar --to-command='
case "$TAR_FILENAME" in
  *.tar)
    mkdir -p "$SCRATCH_DIR/ILSVRC2012/train/${TAR_FILENAME%.tar}"
    tar xf - -C "$SCRATCH_DIR/ILSVRC2012/train/${TAR_FILENAME%.tar}"
    ;;
esac
'

cd -
