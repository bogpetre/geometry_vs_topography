# Geometry vs Topography
This repo contains code for "Cortical maps diverge, representations converge along cortical hierarchy"

## Setup

This setup and downloading of a minimal dataset will require ~1-2 hours, but your mileage may vary 
depending on internet speeds and *nix proficiency.

It is highly recommended that you deploy a conda environment for testing this code.

```
conda create env -n geometry_vs_topography --python=3.9
```

Now install some necessary C++ libraries

```
conda install -c conda-forge libblas eigen lapack nlohmann_json
```

For all analyses except training neural nets, go to the same level of the
directory tree as where this README file is, and invoke the following 
to compile rdm_similarty and install python packages

```
pip install -r requirements.txt
pip install .
make
```

You will also need to install/aquire the additional non-python dependencies below. In 
particular, neuromaps (installed by requirements.txt) and the nipype pipelines in
scripts/ expect wb_command (Dependences: connectome workbench) to be on your system 
path,

```
PATH=$PATH:<connectome_workbench_binary_directory>
```

Finally, copy config_template.json to config.json and update the paths within to
point to the location of your copy of the necessary dependencies (below). Note that if
you do not have access to the restricted HCP data you can continue without it. This will
only prevent you from rerunning the heritability analyses (figure 6).

For training neural networks you will need the TDANN library from the neuroAI lab, which
uses a different python version and is best run from a different conda environment. For
details refer to https://github.com/neuroailab/TDANN. you will additionally need
nltk for semantic superseting ImageNet categories. Note that if you need to run this on
more modern hardware (e.g. H200 GPUs), you may need to update pytorch, torchvision and
patch some minor things in VISSL, the self-supervised learning framework that TDANN
uses. Thankfully, none of this should be necessary if you're just using our pretrained
model weights from OSF.

To avoid needing to rerun all analysis scripts (tens of thousands of CPU and GPU hours
worth of compute), you can find the analysis derivatives used to generate the manuscript
tables and figures here: https://osf.io/xmyrn/overview. Refer to the README included
therein for details on which files pertain to which figures.

### Dependencies

Python:
* See requirements.txt file

Matlab:
* canlabCore: https://github.com/canlab/CanlabCore
* cifti matlab libraries: https://github.com/Washington-University/cifti-matlab
* spm12: https://www.fil.ion.ucl.ac.uk/spm/docs/installation/
* Neuroimaging Pattern Masks: https://github.com/canlab/Neuroimaging_Pattern_Masks/
* RSA Toolbox (modified for multirun and multisession data): https://github.com/bogpetre/rsatoolbox_matlab

C/C++:
* libblas
* eigen
* lapack
* nlohmann_json
(conda install -c conda-forge libblas eigen lapack nlohmann_json)

Binaries:
* connectome workbench: https://www.humanconnectome.org/software/get-connectome-workbench
* FSL: https://fsl.fmrib.ox.ac.uk/fsl/docs/#/install/index
* FreeSurer v7.4.1: https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall

Data:
* HCP data: s3://hcp-openaccess/
* HCP restricted data (only for heritability analysis): https://www.humanconnectome.org/study/hcp-young-adult/document/restricted-data-usage
* ImageNet training and validation data (only for ANN modeling): https://image-net.org/

Note, the use of HCP restricted data prohibits us from sharing specific subject ids. This 
prevents us from identifying the exemplary subjects we use for illustrative purposes in the
manuscript. This information was shared with HCP though and is available to users who agree
to the restricted data usage agreement. Once the key is obtained the appropriate subject
dyad can be assigned in the config.json file and figure 2 can be regenerated.

### canlab2024

This repo requires the canlab2024 atlas. Due to licensing restrictions we cannot 
distribute this atlas publically. Instead, it needs to be assembled locally from 
files you download yourself from remote sources that are properly licensed to
distribute the source files. We've provided an assembly script which will automate
this process for you. After downloading the above dependencies add the matlab
dependencies to your path. Add all subdirectories for all repos except spm12,

```
addpath(genpath(<repoPath>))
addpath(<spm12_path>)
```

Now in matlab invoke,

```
atl = load_atlas('canlab2024_coarse_fsl6_2mm');
create_CANLab2024_CIFTI_subctx('MNI152NLin6Asym','coarse',2,atl);
```

Now navigate to wherever you installed Neuroimaging_Pattern_masks and invoke the following
bash script,

```
Atlases_and_parcellations/2024_CANLab_atlas/src/create_CANLab2024_atlas_cifti.sh
```

If this is not possible for you for whatever reason you can instead use openCANLab2024,
which differs in some brainstem nuclei that aren't critical in this projection. You can
find that in the Neuroimaging Pattern Masks repo under:

Atlases_and_parcellations/2024_CANLab_atlas/openCANLab2024_MNI152NLin6Asym_coarse_2mm.dlabel.nii

Update config.json such that the canlab2024 variable points to openCanlab2024 instead of 
CANLab2024

## Usage

Scripts to reproduce first level analysis and evaluate ANNs are available in scripts/. In particular 
note,
* hcp_glm_msmall_grayord_spm.py: fits first level models and estimates geometires using task data
* hcp_dual_regression_msmall_grayord_spm.py: estimates individualized ICA networks and geometries.
* hcp_glm_msmall_grayord_spm_single_blocks.py: fits first level models and estimates nearest centroid classification classification performance
For details on additional scripts refer to scripts/README

These need to be run iteratively over participants. Macros to do this on a SLURM HPC system are
in macros/. It should be simple to adapt these to other job scheduling systems like torque or SGE.
Without an HPC environment it's not practical to run these analyses (figures 3-5 took 2-3 months
worth of CPU hours to produce, figure 6 took much longer). You can however run a single pair of
participants on a PC in a day or so to help understand the underlying processes if that's desired.
Pick a pair from resources/paired_sid.csv if that's the case. By inspecting analysis intermediates 
you can better evaluate each step's technical characteristics. Specific steps for doing this are 
detailed below in the Demo section. To regenerate figures without rerunning everything you can
simply work with the analysis derivatives which are available via osf (https://osf.io/xmyrn/overview).

Once the above are run, topographic similarity measures and geometric similarity 
measures can be computed using pairwise_parcel_op for cosine similarity measures and 
rdm_similarity for efficient and scalable computation of geometric similarities
from the derivatives of the above analyses (example scripts in macros/).

You will also need to generate spin permuted neuromaps maps locally, since they're too big for 
github. This can be done with scripts/get_parcellated_neuromap_vals.py, which is called by
macros/prep_neuromap_data.sh

Once the above scripts have prepared geometric and topographic similarity measures and you have 

For minimal usage examples refer to the Demo section below and subsections of interest.

## Demo

Replicating all analyses would take months of CPU hours, but because most analyses are performed
dyad-wise and aggregated across dyads, an informative replication ca nbe performed on a single
dyad using a "typical" workstation intsead of a high performance computing cluster. The examples
below demonstrate how to do this. These are managed by nipype pipelines, and by inspecting the
working directory and intermediate files you can gain insight into implementation details of the
first level GLM, dual regression approach, spatial whitening operations and RSA.

This demo assumes you have already installed this repository, following the instructions in "Setup"
above.

If you do not want to rerun first level analysis you can simply use the OSF directory to regenerate
main findings from analysis derivatives (https://osf.io/xmyrn/overview). Just refer to the neuromap
prep section below as well.

### System requirements

Linux system (Mac is untested, but may work too)
100G disk space
48G memory
1 CPU

Data has been tested using the following python library versions:

```
channels:
  - conda-forge
  - defaults
dependencies:
  - _openmp_mutex=4.5=20_gnu
  - bzip2=1.0.8=hda65f42_9
  - ca-certificates=2026.5.20=hbd8a1cb_0
  - lapack=3.11.0=8_netlib
  - ld_impl_linux-64=2.45.1=default_hbd61a6d_102
  - libblas=3.11.0=8_h4a7cf45_openblas
  - libexpat=2.8.1=hecca717_1
  - libffi=3.5.2=h3435931_0
  - libgcc=15.2.0=he0feb66_19
  - libgcc-ng=15.2.0=h69a702a_19
  - libgfortran=15.2.0=h69a702a_19
  - libgfortran5=15.2.0=h68bc16d_19
  - libgomp=15.2.0=he0feb66_19
  - liblapack=3.11.0=8_h47877c9_openblas
  - liblzma=5.8.3=hb03c661_0
  - libnsl=2.0.1=hb9d3cd8_1
  - libopenblas=0.3.33=pthreads_h94d23a6_0
  - libsqlite=3.53.2=h0c1763c_0
  - libuuid=2.42.1=h5347b49_0
  - libxcrypt=4.4.36=hd590300_1
  - libzlib=1.3.2=h25fd6f3_2
  - ncurses=6.6=hdb14827_0
  - nlohmann_json=3.12.0=h54a6638_1
  - openssl=3.6.3=h35e630c_0
  - pip=25.2=pyh8b19718_0
  - python=3.9.23=hc30ae73_0_cpython
  - readline=8.3=h853b02a_0
  - setuptools=80.9.0=pyhff2d567_0
  - tk=8.6.13=noxft_h366c992_103
  - wheel=0.45.1=pyhd8ed1ab_1
  - zstd=1.5.7=hb78ec9c_6
  - pip:
      - acres==0.5.0
      - certifi==2026.5.20
      - charset-normalizer==3.4.7
      - ci-info==0.4.0
      - click==8.1.8
      - contourpy==1.3.0
      - cycler==0.12.1
      - eigen==0.1.1
      - etelemetry==0.3.1
      - filelock==3.19.1
      - fonttools==4.60.2
      - geometry-vs-topography==1.0.0
      - h5py==3.14.0
      - hcp-utils==0.1.0
      - idna==3.18
      - importlib-resources==6.5.2
      - isodate==0.7.2
      - joblib==1.5.3
      - kiwisolver==1.4.7
      - looseversion==1.3.0
      - lxml==6.1.1
      - matplotlib==3.9.4
      - networkx==3.2.1
      - neuromaps==0.0.7
      - nibabel==5.3.3
      - nilearn==0.12.1
      - nipype==1.10.0
      - nipype-workbench-extensions==0.1.0
      - nlohmann-json==3.12.0
      - numpy==2.0.2
      - packaging==26.2
      - pandas==2.3.3
      - pillow==11.3.0
      - prov==2.1.1
      - puremagic==1.30
      - pydot==4.0.1
      - pyparsing==3.3.2
      - python-dateutil==2.9.0.post0
      - pytz==2026.2
      - rdflib==7.6.0
      - requests==2.32.5
      - scikit-learn==1.6.1
      - scipy==1.13.1
      - simplejson==4.1.1
      - six==1.17.0
      - stonefish-license-manager==0.7.2
      - threadpoolctl==3.6.0
      - traits==7.1.0
      - typing-extensions==4.15.0
      - tzdata==2026.2
      - urllib3==2.6.3
      - x21==0.5.3
      - zipp==3.23.1

```

### Data download

Note, the minimal data download is not needed to run figures 1-6 off of OSF analysis 
derivatives, or ANN analyses. This section is only relevant for rerunning timeseries analysis 
of task and resting state data.

First, download the data needed for a minimal run. Assuming you have credentails (an aws access 
key ID and secrete access key) assigned to a profile called "hcp" (in ~/.aws/config) you can 
download a minimal pair like so (from the top level of this repo).

```
mkdir -p data
aws s3 --profile hcp sync \
    --exclude="ROIs/*" --exclude="Native/*" --exclude="xfms/*" \
    --exclude="**/*_s4_*" --exclude="**/Phase*" \
    --exclude="**/*native.func.gii" \
    --exclude="*RibbonVolumeToSurfaceMapping/*" \
    --exclude="**/*feat/*" \
    --exclude="*RestingStateStats/*" \
    --exclude="*stats*" \
    --exclude="**/*SBRef*" \
    --exclude="*ica/*" \
    s3://hcp-openaccess/HCP_1200/100307/MNINonLinear \
    data/HCP_1200/100307/MNINonLinear
aws s3 --profile hcp sync \
    --exclude="ROIs/*" --exclude="Native/*" --exclude="xfms/*" \
    --exclude="**/*_s4_*" --exclude="**/Phase*" \
    --exclude="**/*native.func.gii" \
    --exclude="*RibbonVolumeToSurfaceMapping/*" \
    --exclude="**/*feat/*" \
    --exclude="*RestingStateStats/*" \
    --exclude="*stats*" \
    --exclude="**/*SBRef*" \
    --exclude="*ica/*" \
    s3://hcp-openaccess/HCP_1200/992673/MNINonLinear \
    data/HCP_1200//992673/MNINonLinear
```

And make sure your S1200_imaging variable path in config.json points to data/HCP_1200.
This will download more data than is strictly needed, but better to be 
overinclusive than accidentally miss some stray file used somewhere obscure. The
exclusions already cut each participant directory size down from ~50G to 28G.
your neuromaps ready, figures from the paper can be regenerated using matlab scripts found in 
figures/

### Task similarity dyad example

Modify the run script, macros/run_subj_hcp_msmall_grayord_spm.sh, by substituting a viable
local scratch directory for your pipeline intermediates and uncommenting the "trap" to avoid
autodeletion on completion. Replace this block,

```
cleanup() {
    rm -rf $SCRATCH_DIR
}
trap cleanup EXIT
```

With something like

```
SCRATCH_DIR=$OUT_DIR/scratch/$SID/
cleanup() {
    rm -rf $SCRATCH_DIR
}
#trap cleanup EXIT
```

Now set the SLURM_ARRAY_TASK_ID env varibale corresponding to subject pairs 100307 and 992673
and run macros/run_subj_hcp_msmall_grayord_spm.sh. If you have only downloaded these two 
participants these would be 1 and 2. Otherwise these will be whatever order these subject ids 
come out in when you run ls data/HCP_1200 | cat -n. A loop like the following should take care of
it.

```
cd macros/
for i in 1 2; do
    export SLURM_ARRAY_TASK_ID=$i
    ./run_subj_hcp_msmall_grayord_spm.sh
done
```

run_subj_hcp_msmall_grayord_spm.sh will run the first level analysis and prepare inputs for
topographic and geometric similarity analyses. These will take several hours to run. Note that
they must be run from inside the macros directory or relative paths won't point to the right 
locations.

To compute similarity metrics from these outputs we used the the *bsc.sh scripts. To replicate
this for your example dyad you can invoke the following,

```
export SLURM_ARRAY_TASK_ID=1
macros/run_subj_hcp_msmall_grayord_spm_bsc.sh
```

Noe that this also uses a scratch directory, but not for nipype pipelines, and these
are not likely to be very informative. You can use a modification to retain these if you like, 
just as you did by commenting out the trap and redirecting scratch in the 
macros/run_subj_hcp_msmall_grayord_spm.sh scripts however a more useful approach may be to 
simply perform these computations in matlab and step through the code in a debugger. For 
replicating the results from the paper exactly you should use the C++ code as the *bsc.sh 
code does, but C++ isn't as easy to read as matlab in a debugger, and the latter is also made
available by our prototyping scripts as a substitute. For the latter refer to the usage in 
src/rdm_similarity/matlab/unit_test_rdm_cosim_scripts.m

The approach involving the *bsc.sh scripts will produce outputs in 
derivatives/hcp_glm_msmall_grayord_spm/bsc. First level model outputs will be in
derivatives/hcp_glm_msmall_grayord_spm/results/. For example, standard first level model
outputs will be in 
derivatives/hcp_glm_msmall_grayord_spm/results/<SID>/all_tasks/tstats/
You can view theese in wb_view (https://www.humanconnectome.org/software/connectome-workbench).
You will need some FS LR 32k underlay surfaces like the S1200 inflated cortical surface
(https://github.com/rmldj/hcp-utils/tree/master/hcp_utils/data) and a volumetric reference
which you can find from the Neuroimaging_Pattern_Masks repository:
(https://github.com/canlab/Neuroimaging_Pattern_Masks/tree/master/templates/MNI152NLin6Asym_T1_2mm.nii.gz)


### Resting state network similarity dyad example

Modify the run script, macros/run_rsn_hcp_msmall_grayord_spm.sh like you did the GLM script 
above, by substituting a viable local scratch directory for your pipeline intermediates and 
uncommenting the "trap" to avoid autodeletion on completion. Replace this block,

```
cleanup() {
    rm -rf $SCRATCH_DIR
}
trap cleanup EXIT
```

With something like

```
SCRATCH_DIR=$OUT_DIR/scratch/$SID/
cleanup() {
    rm -rf $SCRATCH_DIR
}
#trap cleanup EXIT
```

Now set the SLURM_ARRAY_TASK_ID env varibale corresponding to subject pairs 100307 and 992673
and run macros/run_subj_hcp_msmall_grayord_spm.sh (refer to the GLM analysis above for details).
A loop like the following should take care of running the first level analyses.

```
cd macros/
for i in 1 2; do
    export SLURM_ARRAY_TASK_ID=$i
    ./run_rsn_hcp_msmall_grayord_spm.sh
done
```

run_rsn_hcp_msmall_grayord_spm.sh will run the first level analysis (dual regression) and prepare 
inputs for topographic and geometric similarity analyses. These will take several hours to run, 
same as the task scripts. Note once again that they must be run from inside the macros/ directory 
for relative paths to be correct.

To compute similarity metrics from these outputs we used the the *bsc.sh scripts. To replicate
this for your example dyad you can invoke the following,

```
export SLURM_ARRAY_TASK_ID=1
macros/run_rsn_hcp_msmall_grayord_spm_bsc.sh
```

You can also use the matlab prototyping script as a substitute. Refer to the task demo above
for details on this.

The approach involving the *bsc.sh scripts will produce outputs in 
derivatives/restingstate/hcp25/bsc. First level model outputs will be in
derivatives/restingstate/hcp25/results/. For example, standardized ICA maps (tstats) 
will be in 
derivatives/restingstate/hcp25/results/100307/standardized_betas/cifti_math_results.dscalar.nii
You can view theese in wb_view (https://www.humanconnectome.org/software/connectome-workbench).
The approach is the same as for the task results.


### Neuromap Prep

To perform inference over spatial maps you will need null models for spin tests. These can be
obtained by running scripts/neuromaps get_parcellated_neuromap_vals.py. The minimum reqs above
list 1 CPU. If you have fewer than 12 available modify this script will automatically use them
though. If you don't want this modify the macro accordingly.

```
cd macros/
./prep_neuromap_data.sh
```

### Replicating figures 1-6

To replicate these figures make sure you've downloaded and extracted hcp_glm_msmall_grayord_spm/bsc.tar.bz 
(a bz2 archive, despite the extension), and hcp_glm_msmall_grayord_spm/results.tar.bz2. 
hcp_glm_grayord_spm.tar.bz2 (for replication of non-msmall results), 
hcp_glm_msmall_grayord_spm_gordon.tar.bz2 (for replication with the Gordon atlas), models.tar.bz2 
(for ANNs), retest.tar.bz2 (for figure 6) and single_blocks_msm_grayord_spm.tar.bz2 (for brain decoding).
Of these the most important is hcp_glm_msmall_grayord_spm (for figures 2-5 and most of 6).

Additionally for figures 3-5 make sure you've run the Neuromap Prep detailed above.

You should now be able to run the matlab scripts in figures/ to reproduce the plots from the manuscript.
