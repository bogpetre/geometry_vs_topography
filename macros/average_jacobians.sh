#!/bin/bash

set -x

SCRATCH_DIR=$TMPDIR/$(uuidgen)
cleanup() {
    rm -rf $SCRATCH_DIR
}
trap cleanup EXIT
mkdir -p $SCRATCH_DIR


HCP_DATA=$(cat ../config.json | \
    python3 -c "import sys, json; print(json.load(sys.stdin)['hcp_participant_data']['S1200_imaging'])")

sid_list=($(cat ../resources/paired_sid.csv | awk -F, '{print $1"\n"$2}'))

for sid in ${sid_list[@]}; do
    LRfiles=($(ls /dartfs/rc/lab/D/DBIC/DBIC/archive/HCP/HCP1200/$sid/MNINonLinear/Results/tfMRI_*_LR/*Jacobian.nii.gz))
    RLfiles=($(ls /dartfs/rc/lab/D/DBIC/DBIC/archive/HCP/HCP1200/$sid/MNINonLinear/Results/tfMRI_*_RL/*Jacobian.nii.gz))

    masks=($(ls /dartfs/rc/lab/D/DBIC/DBIC/archive/HCP/HCP1200/$sid/MNINonLinear/Results/tfMRI_*/brainmask_fs.2.nii.gz))

    if [ ! ${#LRfiles[@]} -eq 7 ] || [ ! ${#RLfiles[@]} -eq 7 ]; then
        echo "$sid is missing Jacobian files"
    else
        fslmerge -t $SCRATCH_DIR/${sid}_LR_ts.nii.gz ${LRfiles[@]} &
        fslmerge -t $SCRATCH_DIR/${sid}_RL_ts.nii.gz ${RLfiles[@]} &
        fslmerge -t $SCRATCH_DIR/${sid}_mask_ts.nii.gz $masks &

        wait

        fslmaths $SCRATCH_DIR/${sid}_mask_ts.nii.gz -Tmax $SCRATCH_DIR/${sid}_mask.nii.gz &

        fslmaths $SCRATCH_DIR/${sid}_LR_ts.nii.gz -add 0.00001 -log -Tmean $SCRATCH_DIR/${sid}_LR.nii.gz &
        fslmaths $SCRATCH_DIR/${sid}_RL_ts.nii.gz -add 0.00001 -log -Tmean $SCRATCH_DIR/${sid}_RL.nii.gz &

        wait
    fi
done

fslmerge -t $SCRATCH_DIR/mask_ts.nii.gz $SCRATCH_DIR/*_mask.nii.gz &
fslmerge -t $SCRATCH_DIR/LR_ts.nii.gz $SCRATCH_DIR/*_LR.nii.gz &
fslmerge -t $SCRATCH_DIR/RL_ts.nii.gz $SCRATCH_DIR/*_RL.nii.gz &

wait

fslmaths $SCRATCH_DIR/mask_ts.nii.gz -Tmax $SCRATCH_DIR/mask.nii.gz

fslmaths $SCRATCH_DIR/LR_ts.nii.gz -Tmean -mas $SCRATCH_DIR/mask.nii.gz ../derivatives/LR_Jacobian.nii.gz &
fslmaths $SCRATCH_DIR/RL_ts.nii.gz -Tmean -mas $SCRATCH_DIR/mask.nii.gz ../derivatives/RL_Jacobian.nii.gz &

wait
