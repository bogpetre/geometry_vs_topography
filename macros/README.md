The purpose of macros/ scripts is as the main entry point for population derivatives/ or as
a source for examples on how to use other scripts in this library to populate derivatives/.
In most cases macros/ calls on lower level scripts and programs from the library, like those 
in scripts/ or bin/, but regardless its purpose is high level. Every script in here should
produce derivatives/ outputs that are directly used by scripts in figures/.

Scripts use relative paths, mainly to query config.json, and are likely to break if you move 
them.

Descriptions:
* prep_neuromap_data.sh - generate spin permuted neuromaps for spatial inference (Figures 3-5, S5-S9)
* run_ann_model_similarity_eval.sh - invokes scripts/eval_ann_model_similarities.py to compute cosine similarity and CKA metrics
* run_rsn_hcp_msmall_grayord_spm.sh - main script for invoking dual regression and resting state RDMs generation script. Used in Figure 5 and 6.
* run_rsn_hcp_msall_grayord_spm.sh - compute cosine similarity and representational similarity measures for resting state networks for dyads of unrelated participants (figure 5)
* run_rsn_hcp_msmall_grayord_spm_bsc_full.sh - invokes similarity evals for all pairwise combinations of participants (for heritability analysis, figure6)
* run_rsn_hcp_msmall_grayord_spm_bsc_shuffled.sh - computes similarity measures using 100 different permutations of networks to produce null reference line in figure 5D
* run_rsn_hcp_msall_grayord_spm_retest.sh - similar to run_rsn_hcp_msmall_grayord_spm.sh but evaluates follow-up visit data instead of main visit data.
* run_rsn_hcp_msmall_grayord_spm_retest_selfsimilarity.sh - computes true RSN test-retest reliability (across visits) to use as a noise ceiling reference in heritability analyses
* run_single_blk_small_grayord_spm.sh - similar to run_subj_hcp_msmall_grayord_spm.sh but instead of treating each task condition as IVs, this treats each unique stimulus/response block as a unique IV, and trains nearest centroid classifiers to discriminate all resulting pairwise task combinations. Results are used in figure 2.
* run_subj_hcp_grayord_spm.sh - similar to run_subj_hcp_msmall_grayord_spm.sh exceut using anatomically aligned data. Used for figure S5.
* run_subj_msmall_hcp_grayord_spm.sh - main analysis script. Performs task GLMs, and computes regional RDMs. Used in figures 3-4 and 6.
* run_subj_msmall_hcp_grayord_spm_bsc.sh - compute cosine and RDM similarities for dyads of unrelated participants (Figures 3-4).
* run_subj_msmall_hcp_grayord_spm_bsc_full.sh - compute cosine and RDM similarities for all pairs of participants (heritability analysis, Figure 6).
* run_subj_msmall_hcp_grayord_spm_bsc_shuffled.sh - compute permutation-null similarities for reference line in figure 3D.
* run_subj_msmall_hcp_grayord_spm_gordon.sh - similar to run_subj_msmall_hcp_grayord_spm.sh except using a different parcellation (figure S6).
* run_subj_msmall_hcp_grayord_spm_retest.sh - similar to run_subj_msmall_hcp_grayord_spm.sh except using follow-up visit data.
* run_subj_msmall_hcp_grayord_spm_retest_selfsimilarity.sh computes true task test-retest reliability (across visits) to use as a noise ceiling reference in heritability analyses.
* train_tdann.sh - invokes TDANN repo training scripts (from the neuroAI lab) to train 14 TDANNs and 14 ResNet models from 14 different seeds.
