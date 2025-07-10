
### RDM similarity estimation

The program in this folder implements an efficient algorithm for computing RDM similarites
across a brain parcellation. It assumes you have RDMs for each of many regions, and that for
each RDM you have a unique covariance matrix to use for whitening. These are produced by
the nipype pipelines provided in the scripts/ folder of this repository and this program
is designed to work with the whitening_matrix_out.bin, whitening_matrix_out.json and 
[euclidean|standardized|whitened|]_distance.csv files produced by these pipelines.

rdm_similarity reads in RDM covariance matrices, pools them across RDMs under consideration 
and whitens RDM distances by the inverse of this matrix. The program by default only computes
similarites for matched regions, but optionally it can also compute similarities of 
geometries across all region-region pairs, a much more computatoinally demanding operation, 
but nevertheless tractable even for quite large matrices (e.g. 100 x 100 task RDMs) thanks
to this program.

## INSTALLATION:

Dependencies:
lapack
eigen
nlohmann_json

conda-forge -c libblas eigen nlohmann_json

make

If there are problems compiling the matlab/ subfolder contains prototyping code that can
be used instead. This is about 2.5x slower, but otherwise performs the same calculations.
Refer to matlab/unit_test_rdm_cosim_scripts.m for an idea of how to use this code.

## USAGE:

./rdm_similarity --help

The algorithm is designed to work with potentially very large RDMs where the covariance
matrices (which scale O(n^4) in tasks) may be many GB. Consequently the whitening matrices 
are not all read into memory at once. These can be quite large and easily swamp the memory 
capacity of most HPC compute nodes or PCs (macs included). Instead they're read in on-demand. 
This can incur a substantial speed penalty if these read operations occur on a network file 
system (which is commonly the case on HPC systems). Instead you should copy these files to 
local scratch space, which exists on most modern HPC compute nodes precisely for this reason. 
Otherwise you will see a 10x speed penalty rather than a 2.5x speed benefit from using this 
program.
