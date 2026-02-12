# Geometry vs Topography
This repo contains code for "Cortical maps diverge, representations converge along cortical hierarchy"

## Setup

From all analyses except training neural nets, go to the same level of the
directory tree as where this README file is, and invoke the following 
to compile rdm_similarty and install python packages

```
pip install -r requirements.txt
pip install .
make
```

If you get compilation problems due to missing libraries try,

```
conda install -c conda-forge libblas eigen lapack nlohmann_json
```

and then rerun make.

You will also need to install/aquire the additional non-python dependencies below. In 
particular, neuromaps (installed by requirements.txt) and the nipype pipelines in
scripts/ expect wb_command (Dependences: connectome workbench) to be on your system 
path,

```
PATH=$PATH:<connectome_workbench_binary_directory>
```

Finally, copy config_template.json to config.json and update the paths within to
point to the location of your copy of the necessary dependencies (below).

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
you can better evaluate each step's technical characteristics.

Once the above are run, topographic similarity measures and geometric similarity 
measures can be computed using pairwise_parcel_op for cosine similarity measures and 
rdm_similarity for efficient and scalable computation of geometric similarities
from the derivatives of the above analyses (example scripts in macros/).

You will also need to generate spin permuted neuromaps maps locally, since they're too big for 
github. This can be done with scripts/get_parcellated_neuromap_vals.py, which is called by
macros/prep_neuromap_data.sh

Once the above scripts have prepared geometric and topographic similarity measures and you have 
your neuromaps ready, figures from the paper can be regenerated using matlab scripts found in 
figures/
