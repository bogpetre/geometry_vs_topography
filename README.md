# geometry_vs_topography
This repo contains code for "Common representations underlie idiosyncratic neural topographies"

# Setup

### Dependencies

Matlab:
* canlabCore: https://github.com/canlab/CanlabCore
* cifti matlab libraries: https://github.com/Washington-University/cifti-matlab
* spm12: https://www.fil.ion.ucl.ac.uk/spm/docs/installation/
* Neuroimaging Pattern Masks: https://github.com/canlab/Neuroimaging_Pattern_Masks/

Python:
* nipype_workbench_ext: https://github.com/bogpetre/nipype_workbench_ext

Binaries:
* connectome workbench: https://www.humanconnectome.org/software/get-connectome-workbench

Data:
* HCP data: s3://hcp-openaccess/
* HCP restricted data (only for heritability analysis): https://www.humanconnectome.org/study/hcp-young-adult/document/restricted-data-usage

Note, the use of HCP restricted data prohibits us from sharing specific subject ids. This 
prevents us from identifying the exemplary subjects we use for illustrative purposes in the
manuscript. This information was shared with HCP though and is available to users who agree
to the restricted data usage agreement. Once the key is obtained the appropriate subject
dyad can be assigned in the config.json file and figure 1 can be regenerated.

### canlab2024

This repo requires the canlab2024 atlas. Due to licensing restrictions we cannot 
distribute this atlas publically. Instead, it needs to be assembled locally from 
files you download yourself from remote sources that are properly licensed to
distribute the source files. We've provided an assembly script which will automate
this process for you. After downloading the above dependencies add the matlab
dependencies to your path. Add all subdirectories for all repos except spm12,

addpath(genpath(<repoPath>))
addpath(<spm12_path>)

Now in matlab invoke,

atl = load_atlas('canlab2024_coarse_fsl6_2mm');
create_CANLab2024_CIFTI_subctx('MNI152NLin6Asym','coarse',2,atl);

Now navigate to wherever you installed Neuroimaging_Pattern_masks and invoke the following
bash script,

Atlases_and_parcellations/2024_CANLab_atlas/src/create_CANLab2024_atlas_cifti.sh

If this is not possible for you for whatever reason you can instead use openCANLab2024,
which differs in some brainstem nuclei that aren't critical in this projection. You can
find that here:

Atlases_and_parcellations/2024_CANLab_atlas/openCANLab2024_MNI152NLin6Asym_coarse_2mm.dlabel.nii

Update config.json such that the canlab2024 variable points to openCanlab2024 instead of 
CANLab2024
