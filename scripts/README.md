Files in scripts/ make use of this library, but do not need to be directly integrated into 
the library (i.e. src/geometry_vs_topography) because they're high level. By keeping them 
as stand alone scripts they can be more easily tweaked and modified on-demand for customized 
analysis pipelines. They can be used directly, but in most cases are still general purpose 
enough that they need to be invoked in particular ways to regenerate the analyses of the 
accompanying study. macros/ contains examples of the particular invocations needed.

* ann_classification_metrics.py - returns performance and agreement metrics for TDANNs and ResNets
* ann_diagram.ipynb - generates panels 1A and 1B (example TDANN topographies and representations)
* average_jacobians.sh - computes mean study-wide SDC displacements (supplemental figure 2)
* copy imagenet_train_to_scratch.sh - a utility function to copy and extract training data to a local SSD storage device. Useful if training data is stored on a (slower) network filesystem.
* eval_ann_model_similarities.py - computes cosine similarity and CKA for neural network models (figure 1C)
* get_parcellated_neuromap_vals.py - downloads and parcellates neuromaps using a specified atlas (an HCP91k CIFTI dlabel file). Generates (spin) permuted maps if desired.
* hcp_dual_regression_msmall_grayord_spm.py - Estimates topographies and geometries of RSNs for a specified HCP participant
* hcp_glm_msmall_grayord_spm.py - Estimates topographies and geometries of task evoked responses for a specified HCP participant
* hcp_glm_msmall_grayord_spm_single_blocks.py - Estimates single block contrasts for each task and estimates classification performance of binary classifiers for a specified HCP participant
