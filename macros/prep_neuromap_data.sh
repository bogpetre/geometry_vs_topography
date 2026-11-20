#!/bin/bash
# This script generates parcellated neuromaps using CANLab2024 and the Gordon atlas as well as their
# null (spin) permuted versions. The latter takes time (hours) to run. Update NCPUs variable in the
# script header and run. config.json must also be correctly configured in the parent directory.

set -x

NCPUs= # substitute with number of CPUs

# auto detect CPU count if not specified
n_cpu_avail=$(cat /proc/cpuinfo  | grep processor | wc -l)
if [ "x$n_cpu_avail" != "x" ]; then
    if [ "x$NCPUs" == "x" ]; then
        if [ $n_cpu_avail -gt 12 ]; then
            NCPUs=12
        else
            NCPUs=$n_cpu_avail
        fi
    fi
fi

CANLab2024=$(readlink -f $(cat ../config.json  | grep canlab2024 -A1 | grep path | awk '{print $2}' | sed 's/"//g'))
Gordon=$(readlink -f ../resources/Gordon_333Cort.32k.dlabel.nii)

if [ "x$CANLab2024" == "x" ]; then echo "Could not find CANLab2024. Please manually specify path in script header"; fi
if [ "x$Gordon" == "x" ]; then echo "Could not find Gordon Atlas. Please manually specify path in script header"; fi

echo "Computing mean neuromap values in CANLab2024"
mkdir -p ../resources/neuromaps/canlab2024_parcel_vals/
python3 -u ../scripts/get_parcellated_neuromap_vals.py \
    --out ../resources/neuromaps/canlab2024_parcel_vals/ \
    --atlas $CANLab2024 --cpus $NCPUs

echo "Computing mean neuromap values in Gordon atlas"
mkdir -p ../resources/neuromaps/gordon_parcel_vals/
python3 -u ../scripts/get_parcellated_neuromap_vals.py \
    --out ../resources/neuromaps/gordon_parcel_vals/ \
    --atlas $Gordon --cpus $NCPUs

echo "Computed permuted (spun) neuromap null values in CANLab2024 (this is slow, order of hours)"
mkdir -p ../resources/neuromaps/canlab2024_permuted_annotations/
python3 -u ../scripts/get_parcellated_neuromap_vals.py \
    --out ../resources/neuromaps/canlab2024_permuted_annotations/ \
    --atlas $CANLab2024 --cpus $NCPUs --nperms 5000

echo "Computed permuted (spun) neuromap null values in Gordon atlas (this is slow, order of hours)"
mkdir -p ../resources/neuromaps/gordon_permuted_annotations/
python3 -u ../scripts/get_parcellated_neuromap_vals.py \
    --out ../resources/neuromaps/gordon_permuted_annotations/ \
    --atlas $Gordon --cpus $NCPUs --nperms 5000
