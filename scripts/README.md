Files in scripts/ make use of this library, but do not need to be directly integrated into 
the library because they're high level. By keeping them as stand alone scripts they can
be more easily tweaked and modified on-demand for customized analysis pipelines. They can
be used directly, but in most cases are still general purpose enough that they need to be
invoked in particular ways to regenerate the analyses of the accompanying study. macros/
contains examples of the particular invocations needed.

* get_parcellated_neuromap_vals.py - downloads and parcellates neuromaps using a specified atlas (an HCP91k CIFTI dlabel file). Generates (spin) permuted maps if desired.
* hcp_dual_regression_msmall_grayord_spm.py - Estimates topographies and geometries of RSNs for a specified HCP participant
* hcp_glm_msmall_grayord_spm.py - Estimates topographies and geometries of task evoked responses for a specified HCP participant
* hcp_glm_msmall_grayord_spm_single_blocks.py - Estimates single block contrasts for each task and estimates classification performance of binary classifiers for a specified HCP participant
