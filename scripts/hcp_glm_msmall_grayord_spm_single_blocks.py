# code adapted from
# https://nipype.readthedocs.io/en/latest/users/examples/fmri_fsl.html
#
# This version redoes the hcp_glm version, except now each event block is saved
# independently, no RSA is performed and no task level contrasts are estimated,
# only run level. The outputs are suitable for classification analysis.
#
# our general strategy here will be to take a list of inputs (one for each run, so 
# potentially spanning directions, tasks and subjects), generate 
# "subject/session" info for each and then merge them within  task (so merging 
# direction=['LR', 'RL']) in the firstlevel workflow. This results in a
# list of session_infos from deignspec being submitted to level1design
# which leve1design then automaticaly processes as separate sessions in
# the same GLM (so GLM design is a block diagonal matrix). Since we have no
# task level effects (only run level) it makes no difference that we run them
# jointly rather than as separate GLMs, but this version required less modification
# of the hcp_glm script from which it was derived.
#
# As in the hcp_glm pipeline, we compute post hoc t-stats using the betaToTstat interface
# below. We follow the approach described in 
# https://www.fil.ion.ucl.ac.uk/spm/doc/books/hbf2/pdfs/Ch8.pd
# which if used with contrast codes will reproduce SPM's t-stats.
#
# Note to self: derived from single_blocks_msmall_grayord_spm.py

from __future__ import print_function
from __future__ import division
from builtins import str
from builtins import range

import os  # system functions
import sys
import argparse
import numpy as np
import json

import nipype.interfaces.io as nio  # Data i/o
import nipype.interfaces.fsl as fsl  # fsl
import nipype.interfaces.spm as spm  # neuroimaging library we use for GLM
import nipype.pipeline.engine as pe  # pypeline engine
import nipype.interfaces.utility as util  # utility
import nipype.algorithms.modelgen as model  # model generation

from geometry_vs_topography.nipype.nearestcentroidclf import NearestCentroidClf

# this next block enables multithreading in matlab, which nipype disables by default,
# but massively increases speed of matrix math. Most of our matlab commands are also 
# called after all iterables have converged, so even if with a multithreaded nipype
# workflow this still shouldn't lead to any oversubscription.
from nipype import config, logging
cfg = dict(execution={'single_thread_matlab': False})
config.update_config(cfg)          # must be called before you create nodes
logging.update_logging(config)     # keeps Nipype’s logger in sy

import nipype.interfaces.matlab as mlab
mlab.MatlabCommand.set_default_matlab_cmd("matlab -nodesktop -nosplash")

# load the necessary config file paths for matlab
early_parser = argparse.ArgumentParser()
early_parser.add_argument('--config', type=str, required=True)
args, _ = early_parser.parse_known_args()

with open(args.config) as f:
    config = json.load(f)

mlab.MatlabCommand.set_default_paths([config['matlab_libraries']['spm12'],
                                        config['matlab_libraries']['rsatoolbox'],
                                        os.path.join(config['matlab_libraries']['canlabCore'],'CanlabCore/diagnostics/'),
                                        os.path.join(config['matlab_libraries']['canlabCore'],'CanlabCore/Visualization_functions'),
                                        os.path.join(config['matlab_libraries']['canlabCore'],'CanlabCore/OptimizeDesign11/core_functions/'),
                                        os.path.join(config['matlab_libraries']['canlabCore'],'CanlabCore/Statistics_tools/'),
                                        os.path.join(config['matlab_libraries']['canlabCore'],'CanlabCore/Data_processing_tools/'),
                                        os.path.join(config['matlab_libraries']['canlabCore'],'CanlabCore/Misc_utilities/'),
                                        os.path.join(os.path.dirname(os.path.abspath(args.config)),
                                                    config['matlab_libraries']['custom'])])

# Without this hack this script tends to hang when run over SLURM on NSF filesystems
from nipype.interfaces.spm import SPMCommand
SPMCommand.version = "12.7777"  # any dummy version string
v = SPMCommand().version

# HCP style surface preprocessing
from geometry_vs_topography.glm.preproc import preproc_surf_hcp
# an interface to the rsatoolbox_matlab repo's spatial whitening tools
from geometry_vs_topography.nipype.rsa import SpatialWhitening

# compute VIFs from SPM.mat using canlabCore tools
from geometry_vs_topography.nipype.glm import VIFs, betaToTstat

# available at github.com/bogpetre/nipype_workbench_ext
from nipype_workbench_ext import cifti as wb_cifti
from nipype_workbench_ext import metric as wb_metric
from nipype_workbench_ext import misc as wb_misc

# top level HCP AWS directory
#data_dir = os.path.abspath('/dartfs/rc/lab/D/DBIC/DBIC/archive/HCP/')

# temporal filter and TR length in seconds
hp_cutoff = 200
TR = 0.72

###################################################
# Define functions and classes for subsequent use #
###################################################

from nipype.interfaces.matlab import MatlabCommand
from nipype.interfaces.base import (
    TraitedSpec,
    traits,
    BaseInterface,
    BaseInterfaceInputSpec,
    File,
)
from traits.api import List
import os
from string import Template

# utility function for dealing with MapNode outputs
def pickfirst(files):
    if isinstance(files, list):
        return files[0]
    else:
	    return files


# ########################## #
# Run specific configuration #
# ########################## #


infosource = pe.Node(
    interface=util.IdentityInterface(fields=['subject_id']), name="infosource")

tasksource = pe.Node(
    interface=util.IdentityInterface(fields=['task']), name="tasksource")

directionsource = pe.Node(
    interface=util.IdentityInterface(fields=['direction']), name="directionsource")
    
datasource = pe.Node(
    interface=nio.DataGrabber(
        infields=['subject_id', 'task', 'direction'], 
        outfields=['func_vol', 'func_surf', 
                   'surf_left', 'shape_left', 
                   'surf_right', 'shape_right', 
                   'seg', 'motion']),
    name='datasource')
#datasource.inputs.base_directory = os.path.join(data_dir, 'HCP1200/')
datasource.inputs.template='*'
datasource.inputs.field_template={'func_vol': '%s/MNINonLinear/Results/tfMRI_%s_%s/tfMRI_%s_%s.nii.gz', # this isn't actually used but is useful if you modify this script to correct for confounds
                            'func_surf': '%s/MNINonLinear/Results/tfMRI_%s_%s/tfMRI_%s_%s_Atlas_MSMAll.dtseries.nii',
                            'surf_left': '%s/MNINonLinear/fsaverage_LR32k/%s.L.midthickness.32k_fs_LR.surf.gii',
                            'shape_left': '%s/MNINonLinear/fsaverage_LR32k/%s.L.atlasroi.32k_fs_LR.shape.gii',
                            'surf_right': '%s/MNINonLinear/fsaverage_LR32k/%s.R.midthickness.32k_fs_LR.surf.gii',
                            'shape_right': '%s/MNINonLinear/fsaverage_LR32k/%s.R.atlasroi.32k_fs_LR.shape.gii',
                            'seg': '%s/MNINonLinear/aparc+aseg.nii.gz',
                            'motion': '%s/MNINonLinear/Results/tfMRI_%s_%s/Movement_Regressors.txt'}
datasource.inputs.template_args = {'func_vol': [['subject_id','task','direction',
                                                 'task','direction']],
                                   'func_surf': [['subject_id','task','direction',
                                                  'task','direction']],
                                   'surf_left': [['subject_id','subject_id']],
                                   'shape_left': [['subject_id','subject_id']],
                                   'surf_right': [['subject_id','subject_id']],
                                   'shape_right': [['subject_id','subject_id']],
                                   'seg': [['subject_id']],
                                   'motion': [['subject_id','task','direction']]}
datasource.inputs.sort_filelist = True


datasink = pe.Node(
    interface=nio.DataSink(),
    name="datasink")

datasink.inputs.regexp_substitutions = [
    # this assumes that you prefix your datasink outputs with a results folder in every case
    
    # e.g. results/whitened_betas/_subject_id_100307_task_EMOTION/cifti_math-results.dscalar.nii -> 
    #   results/_subject_id_100307_task_EMOTION/whitened_betas/cifti_math-results.dscalar.nii
    #(r'results/([a-z_]+?)/([\w/]+)/', r'results/\2/\1/'),
    (r'results/([a-z_/]+)/([\w/]+)/', r'results/\2/\1/'),
    
    (r'results/(.*)_subject_id_([\d]+)(.*)', r'results/\2/\1/\3'),
    
    (r'results/(.*)_task_(\w+?)([_/]{1}.*)', r'results/\1/\2/\3'),
    (r'results/(.*)_direction_(\w+?)([_/]{1}.*)', r'results/\1/\2/\3/'),
    (r'_0\.',r'_LR.'),
    (r'_1\.',r'_RL.'),
    (r'_run(\w*)2cifti0', r'LR/\1'),
    (r'_run(\w*)2cifti1', r'RL/\1'),
    (r'_(\w*)2cifti0', r'LR/\1'),
    (r'_(\w*)2cifti1', r'RL/\1'),
    
]


# ###################### #
# Preprocessing Workflow #
# ###################### #
preproc = preproc_surf_hcp(hp_cutoff, TR)

# ########################## #
# firstlvl modeling workflow #
# ########################## #


# task specific event configuration

def runinfo(subject_id, task, direction, data_dir):
    from geometry_vs_topography.glm.designs import block_events
    from nipype.interfaces.base import Bunch
    from copy import deepcopy

    names, onsets, dur = block_events(subject_id, task, direction, data_dir)

    output = Bunch(conditions=names,
                    onsets=deepcopy(onsets),
                    durations=deepcopy(dur),
                    amplitudes=None,
                    tmod=None,
                    pmod=None,
                    regressor_names=None,
                    regressors=None)

    return output, names

runinfo_node = pe.Node(util.Function(input_names=['subject_id', 'task', 'direction', 'data_dir'],
                                 output_names=['run_info','contrast_names'],
                                 function=runinfo),
                        name='runinfo_node')


modelfit = pe.Workflow(name='modelfit')

inputnode_modelfit = pe.Node(
    interface=util.IdentityInterface(fields=[
        'subject_id','task','direction',
        'func']),
    name='inputspec')

designspec = pe.Node(
    interface=model.SpecifySPMModel(), 
    name="designspec")

joinRuns = pe.JoinNode(util.IdentityInterface(
        fields=['run_info','session_info','cifti_template']),
    joinsource='directionsource',
    joinfield=['run_info','session_info','cifti_template'],
    name='joinruns')

level1design = pe.Node(interface=spm.Level1Design(
        model_serial_correlations='FAST',
        global_intensity_normalization='none'), # already done FSL style (media = 10000) by HCP minimal preproc
    name="level1design")
    
cifti2nifti = pe.Node(
    interface=wb_cifti.CiftiConvertNifti(
        smaller_dims=True),
    iterfield=['cifti_in'],
    name='cifti2nifti')
    
lsurfdilate = pe.Node(
    interface=wb_metric.MetricDilate(
        distance=50,
        nearest=True),
    name='lsurfdilate')
    
rsurfdilate = pe.Node(
    interface=wb_metric.MetricDilate(
        distance=50,
        nearest=True),
    name='rsurfdilate')


modelestimate = pe.Node(
    interface=spm.EstimateModel(estimation_method={'Classical': 1}),
    name='modelestimate')
# betatotstat uses the SPM.mat file to pull inputs, so the workflow won't
# know which files it needs from the SPM directory. This ensures necessary
# files aren't naively deleted.
modelestimate.config = {'execution': {'remove_unnecessary_outputs': False}}
    
vifs = pe.Node(
    interface=VIFs(events_only=True),
    name='vifs')

betatotstat = pe.Node(
    interface=betaToTstat(),
    name = 'betatotstat'
)

def select_betas_of_interest(beta_images, run_info):
    # this assumes equal number of contrasts in each session
    import numpy as np

    n_sess = len(run_info)
    
    #spm stacks session intercepts at the end, so we need to 
    # subract them from the count before dividing
    beta_per_sess = (len(beta_images)-n_sess)/n_sess
    
    filt_beta_images = [];
    filt_beta_names = [];
    for i,info in enumerate(run_info):
        ind0 = int(beta_per_sess*i)
        beta_names = set(info.conditions)
        
        # drop Cue condition from Motor task, since it's not of interest (trivial visual stim)
        # drop response and question periods since theyr'e also generic like the motor cue condition
        for name in ['Task-Cue', 'Task-Response', 'Task-Math-Question', 'Task-Story-Question']:
            if name in beta_names:
                beta_names.remove(name)
                
        # drop last trials of emotion task because they overrun the scan duration.
        bname_to_remove = []
        for bname in beta_names:
            if 'Task-EMOTION' in bname and '-05' in bname:
                bname_to_remove.append(bname)
        for bname in bname_to_remove:
            beta_names.remove(bname)
                
        these_beta = [beta_images[ind0 + i] for i,x in enumerate(info.conditions) if x in beta_names]
        these_names = [x for i,x in enumerate(info.conditions) if x in beta_names]
        
        filt_beta_images.append(these_beta)
        filt_beta_names.append(these_names)
    
    # this differs from the hcp_glm script    
    filt_beta_images = sum(filt_beta_images, [])
    filt_beta_names = sum(filt_beta_names, [])
    
    return filt_beta_images, filt_beta_names

selectBetasOfInterest = pe.Node(util.Function(input_names=['beta_images', 'run_info'],
                                                 output_names=['beta_images', 'beta_names'],
                                                 function=select_betas_of_interest),
                                       name='selectbetasofinterest')
                                       

selectTstatsOfInterest = pe.Node(util.Function(input_names=['beta_images', 'run_info'],
                                                 output_names=['tstat_images', 'tstat_names'],
                                                 function=select_betas_of_interest),
                                       name='selecttstatsofinterest')

# these mapnodes map across directions
mergebeta = pe.Node(
    interface=fsl.Merge(
        dimension='t'),
    iterfield=['in_files'],
    name="mergebeta")
    
beta2cifti = pe.Node(
    interface=wb_cifti.NiftiConvertCifti(reset_scalars=True),
    iterfield=['nifti_in'],
    name='beta2cifti')

def makeSetNamesList(names):
    def _makeSetNamesList(names):
        if isinstance(names, list) and isinstance(names[0], list):
            return _makeSetNamesList(names[0])
        else:
            return [(int(i+1), item) for i,item in enumerate(names)]
            
    return _makeSetNamesList(names)

addbetanames = pe.Node(
    interface=wb_misc.SetMapNames(),
    iterfield=['in_file'],
    name="addbetanames")
    

mergeruntstats = pe.Node(
    interface=fsl.Merge(
        dimension='t'),
    iterfield=['in_files'],
    name="mergeruntstats")
    
runtstats2cifti = pe.Node(
    interface=wb_cifti.NiftiConvertCifti(reset_scalars=True),
    iterfield=['nifti_in'],
    name='runtstats2cifti')

addruntstatnames = pe.Node(
    interface=wb_misc.SetMapNames(),
    iterfield=['in_file'],
    name="addruntstatnames")
    
joinTasksMF = pe.JoinNode(util.IdentityInterface(
        fields=['tstats','con']),
    joinsource='tasksource',
    joinfield=['tstats','con'],
    name='jointasks')
    
mergeTstatsAcrossTasks = pe.Node(interface=wb_cifti.CiftiMerge(),
    iterfield=["cifti"],
    name="mergetstatsacrosstasks")
    
mergeContrastsAcrossTasks = pe.Node(interface=wb_cifti.CiftiMerge(),
    iterfield=["cifti"],
    name="mergecontrastsacrosstasks")

modelfit.connect([
    (inputnode_modelfit, runinfo_node, [('subject_id', 'subject_id'),
                                                 ('task','task'),
                                                 ('direction','direction')]),


    # set up SPM design
    (inputnode_modelfit, cifti2nifti, [('func','cifti_in')]),
    (cifti2nifti, designspec, [('out_file', 'functional_runs')]),
    (runinfo_node, designspec, [('run_info','subject_info')]),

    (designspec, joinRuns, [(('session_info', pickfirst), 'session_info')]),
    (joinRuns, level1design, [('session_info', 'session_info')]),


    # run SPM design
    (level1design, modelestimate, [('spm_mat_file', 'spm_mat_file')]),
    (modelestimate, vifs, [('spm_mat_file', 'spm_mat_file')]),


    (inputnode_modelfit, joinRuns, [(('func', pickfirst), 'cifti_template')]),


    # reassemble and merge run-level betas
    (runinfo_node, joinRuns, [(('run_info', pickfirst), 'run_info')]),
    (joinRuns, selectBetasOfInterest, [('run_info', 'run_info')]),
    (modelestimate, selectBetasOfInterest, [('beta_images', 'beta_images')]),
    
    (selectBetasOfInterest, mergebeta, [('beta_images', 'in_files')]),
    (selectBetasOfInterest, addbetanames, [(('beta_names', makeSetNamesList), 'map')]),
    (mergebeta, beta2cifti, [('merged_file', 'nifti_in')]),
    (joinRuns, beta2cifti, [(('cifti_template', pickfirst), 'cifti_template')]),
    (beta2cifti, addbetanames, [('out_file', 'in_file')]),
    
    (addbetanames, joinTasksMF, [(('out_file', pickfirst), 'con')]),
    (joinTasksMF, mergeContrastsAcrossTasks, [('con', 'cifti')]),
    
    
    # make run-level betas into tstats
    (modelestimate, betatotstat, [('spm_mat_file', 'spm_mat_file')]),
    (betatotstat, selectTstatsOfInterest, [('tstats', 'beta_images')]),
    (joinRuns, selectTstatsOfInterest, [('run_info', 'run_info')]),
    
    # merge run-level tstats
    (selectTstatsOfInterest, mergeruntstats, [('tstat_images', 'in_files')]),
    (selectTstatsOfInterest, addruntstatnames, [(('tstat_names', makeSetNamesList), 'map')]),
    (mergeruntstats, runtstats2cifti, [('merged_file', 'nifti_in')]),
    (joinRuns, runtstats2cifti, [(('cifti_template', pickfirst), 'cifti_template')]),
    (runtstats2cifti, addruntstatnames, [('out_file', 'in_file')]),
    
    (addruntstatnames, joinTasksMF, [('out_file', 'tstats')]),
    (joinTasksMF, mergeTstatsAcrossTasks, [('tstats', 'cifti')]),
])

########################
# do spatial whitening #
########################

whiteningwf = pe.Workflow(name='whitening')

inputnode_whitening = pe.Node(
    interface=util.IdentityInterface(fields=[
        'spm_mat_file','atlas', 'run_info']),
    name='inputspec')

atlas2nifti = pe.Node(
    interface=wb_cifti.CiftiConvertNifti(
        smaller_dims=True),
    iterfield=['cifti_in'],
    name='atlas2nifti')
    
joinTaskBetas = pe.JoinNode(util.IdentityInterface(
        fields=['standardized_betas', 'whitened_betas']),
    joinsource='tasksource',
    joinfield=['standardized_betas', 'whitened_betas'],
    name='jointaskbetas')
    
# spatial standardize runwise
stdbetas = pe.Node(
    interface=SpatialWhitening(normmode='runwise', shrinkage=1.0),
    name="stdbetas")
    
splitstdbetas = pe.Node(
    interface=fsl.Split(dimension='t'),
    name="splitstdbetas")

selectStdBetasOfInterest = pe.Node(util.Function(input_names=['beta_images', 'run_info'],
                                                 output_names=['beta_images', 'beta_names'],
                                                 function=select_betas_of_interest),
                                       name='selectstdbetasofinterest')
                                       
mergestdbetas = pe.Node(
    interface=fsl.Merge(
        dimension='t'),
    iterfield=['in_files'],
    name="mergestdbetas")
    
standardizedbeta2cifti = pe.Node(
    interface=wb_cifti.NiftiConvertCifti(reset_scalars=True),
    iterfield=['nifti_in'],
    name='standardizedbeta2cifti')

def makeSetNamesListSubjLevel(names):
    def _makeSetNamesListSubjLevel(names):
        if isinstance(names, list) and isinstance(names[0], list):
            return _makeSetNamesListSubjLevel(names[0])
        else:
            return [(int(i+1), item) for i,item in enumerate(names)]
            
    return _makeSetNamesListSubjLevel(names)

addstdnames = pe.Node(
    interface=wb_misc.SetMapNames(),
    iterfield=['in_file'],
    name="addstdnames")
    
# spatial whitening runwise
whitenbetas = pe.Node(
    interface=SpatialWhitening(normmode='runwise'),
    name="whitenbetas")
    
splitwhitenedbetas = pe.Node(
    interface=fsl.Split(dimension='t'),
    name="splitwhitenedbetas")

selectWhitenedBetasOfInterest = pe.Node(util.Function(input_names=['beta_images', 'run_info'],
                                                 output_names=['beta_images', 'beta_names'],
                                                 function=select_betas_of_interest),
                                       name='selectwhitenedbetasofinterest')
                                       
mergewhitenedbetas = pe.Node(
    interface=fsl.Merge(
        dimension='t'),
    iterfield=['in_files'],
    name="mergewhitenedbetas")
    
whitenedbeta2cifti = pe.Node(
    interface=wb_cifti.NiftiConvertCifti(reset_scalars=True),
    iterfield=['nifti_in'],
    name='whitenedbeta2cifti')

addwhitenednames = pe.Node(
    interface=wb_misc.SetMapNames(),
    iterfield=['in_file'],
    name="addwhitenednames")
    
# merge whitened betas for post-hoc between subject spatial correlation analysis

mergeWhitenedContrastsAcrossTasks = pe.Node(interface=wb_cifti.CiftiMerge(),
    iterfield=['cifti'],
    name="mergewhitenedcontrastsacrosstasks")
    
    
mergeStdContrastsAcrossTasks = pe.Node(interface=wb_cifti.CiftiMerge(),
    iterfield=['cifti'],
    name="mergestandardizedcontrastsacrosstasks")


whiteningwf.connect([
    (inputnode_whitening, atlas2nifti, [('atlas', 'cifti_in')]),
    
    
    # standardize runwise
    (inputnode_whitening, stdbetas, [('spm_mat_file', 'spm_mat_file')]),
    (atlas2nifti, stdbetas, [('out_file', 'atlas')]),
    
    # split std betas by session, merge and convert to LR and RL specific ciftis
    (stdbetas, splitstdbetas, [('whitened_images', 'in_file')]),
    (splitstdbetas, selectStdBetasOfInterest, [
        ('out_files', 'beta_images')]),
    (inputnode_whitening, selectStdBetasOfInterest, [
        ('run_info', 'run_info')]),
        
    (selectStdBetasOfInterest, mergestdbetas, [('beta_images', 'in_files')]),
    (mergestdbetas, standardizedbeta2cifti, [('merged_file', 'nifti_in')]),
    (inputnode_whitening, standardizedbeta2cifti, [('atlas', 'cifti_template')]),
    
    # assign condition names to whitened betas
    (selectStdBetasOfInterest, addstdnames, [(('beta_names', makeSetNamesListSubjLevel), 'map')]),
    (standardizedbeta2cifti, addstdnames, [('out_file', 'in_file')]),
    
    (addstdnames, joinTaskBetas, [('out_file', 'standardized_betas')]),
    (joinTaskBetas, mergeStdContrastsAcrossTasks, [('standardized_betas', 'cifti')]),
    
    
    # whiten runwise
    (inputnode_whitening, whitenbetas, [('spm_mat_file', 'spm_mat_file')]),
    (atlas2nifti, whitenbetas, [('out_file', 'atlas')]),
    
    # split whitened betas by session, merge and convert to LR and RL specific ciftis
    (whitenbetas, splitwhitenedbetas, [('whitened_images', 'in_file')]),
    (splitwhitenedbetas, selectWhitenedBetasOfInterest, [
        ('out_files', 'beta_images')]),
    (inputnode_whitening, selectWhitenedBetasOfInterest, [
        ('run_info', 'run_info')]),
        
    (selectWhitenedBetasOfInterest, mergewhitenedbetas, [('beta_images', 'in_files')]),
    (mergewhitenedbetas, whitenedbeta2cifti, [('merged_file', 'nifti_in')]),
    (inputnode_whitening, whitenedbeta2cifti, [('atlas', 'cifti_template')]),
    
    # assign condition names to whitened betas
    (selectWhitenedBetasOfInterest, addwhitenednames, [(('beta_names', makeSetNamesListSubjLevel), 'map')]),
    (whitenedbeta2cifti, addwhitenednames, [('out_file', 'in_file')]),
    
    (addwhitenednames, joinTaskBetas, [('out_file', 'whitened_betas')]),
    (joinTaskBetas, mergeWhitenedContrastsAcrossTasks, [('whitened_betas', 'cifti')]),    
])

########################################
# Nearest centroid classifier workflow #
########################################

def init_clfwf(name='clf'):
    wf = pe.Workflow(name=name)

    inputnode = pe.Node(
        interface=util.IdentityInterface(fields=[
            'cifti', 'atlas']),
        name='inputspec')

    clf = pe.Node(
        interface=NearestCentroidClf(),
        name='clf')

    outputnode = pe.Node(
        interface=util.IdentityInterface(fields=[
            'clf_perf_csv']),
        name='outputspec')

    wf.connect([
        (inputnode, clf, [('cifti', 'cifti'),
                          ('atlas', 'atlas')]),

        (clf, outputnode, [('out_file', 'clf_perf_csv')]),
    ])

    return wf

stdclfwf = init_clfwf(name='stdclf')
whclfwf = init_clfwf(name='whclf')

################################################################
# Combine preproc, first level and spatial whitening workflows #
################################################################

subjectlevel = pe.Workflow(name="subjectlevel")

subjectlevel.connect([
    (infosource, datasource, [('subject_id', 'subject_id')]),
    (tasksource, datasource, [('task','task')]),
    (directionsource, datasource, [('direction','direction')]),
                             
    (datasource, preproc, [
        ('func_vol', 'inputspec.func_vol'),
        ('func_surf', 'inputspec.func_surf'),
        ('surf_left', 'inputspec.surf_left'),
        ('surf_right', 'inputspec.surf_right')]),
        
    (infosource, modelfit, [('subject_id', 'inputspec.subject_id')]),
    (tasksource, modelfit, [('task','inputspec.task')]),
    (directionsource, modelfit, [('direction','inputspec.direction')]),
    (preproc, modelfit, [('nii2cifti.out_file', 'inputspec.func')]),
        
    (modelfit, whiteningwf, [('modelestimate.spm_mat_file', 'inputspec.spm_mat_file'),
                             ('joinruns.run_info', 'inputspec.run_info'),
                            ]),

    (whiteningwf, stdclfwf, [('mergestandardizedcontrastsacrosstasks.out_file', 'inputspec.cifti')]),
    (whiteningwf, whclfwf, [('mergewhitenedcontrastsacrosstasks.out_file', 'inputspec.cifti')]),
    
    # save desired outputs
    (modelfit, datasink, [('mergecontrastsacrosstasks.out_file', 'results.all_tasks.contrasts'),

                          ('mergetstatsacrosstasks.out_file', 'results.all_tasks.tstats'),
                          
                          ('vifs.vifs', 'results.@vifs'),
                          ('vifs.png', 'results.@png'),
                          ('vifs.hpfilt', 'results.@hpfilt'),
                          
                          ('modelestimate.spm_mat_file', 'results.@spm_mat_file'),
                          ]),
                          
    
    (whiteningwf, datasink, [('mergestandardizedcontrastsacrosstasks.out_file', 'results.all_tasks.standardized_contrasts'),
                             ('mergewhitenedcontrastsacrosstasks.out_file', 'results.all_tasks.whitened_contrasts'),
                            ]),

    (stdclfwf, datasink, [('outputspec.clf_perf_csv', 'results.all_tasks.standardized_contrasts.@clf_perf')]),
    (whclfwf, datasink, [('outputspec.clf_perf_csv', 'results.all_tasks.whitened_contrasts.@clf_perf')]),
])



subjectlevel.inputs.modelfit.designspec.input_units = 'secs'
subjectlevel.inputs.modelfit.designspec.time_repetition = TR
subjectlevel.inputs.modelfit.designspec.high_pass_filter_cutoff = hp_cutoff

subjectlevel.inputs.modelfit.level1design.interscan_interval = TR
subjectlevel.inputs.modelfit.level1design.timing_units = 'secs'
subjectlevel.inputs.modelfit.level1design.bases = {'hrf': {'derivs': [0,0]}}

# ############# #
# main function #
# ############# #

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="GLM estimation of individual blocks of HCP stimuli")
    parser.add_argument('--subject_ids', nargs='*', help="Subject ID. Must match HCP directory name")
    parser.add_argument('--tasks', nargs='*', 
                        default=['EMOTION', 'GAMBLING', 'SOCIAL', 'LANGUAGE', 'RELATIONAL', 'MOTOR', 'WM'],
                        help='Must match HCP task labels which are all capitalized. See default for options.')
    parser.add_argument('--out', type=str, required=True, help='Output path basename. \
                        Outputs will be in subfolders labed by subject_id, task and direction.')
    parser.add_argument('--scratch', type=str, required=True,
                        help='scratch directory where temporary files should be stored')
    parser.add_argument('--n_cpus', type=int, default=int(os.getenv('SLURM_CPUS_PER_TASK',default='1')),
                        help='Number of CPUs available for computation')
    parser.add_argument('--atlas', type=str, required=True,
                        help='cifti dlabels file to use to define whitening parcels')
    parser.add_argument('--data', type=str, required=False,
                        help='Path to HCP data directory immediately above subject folders, e.g. HCP1200')
    parser.add_argument('--config', type=str, required=True, default=os.path.abspath('../config.json'),
                        help='Path to json file containing local environment paths')

    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    if args.data is not None:
        datasource.inputs.base_directory = args.data
        subjectlevel.inputs.modelfit.runinfo_node.data_dir = args.data
    else:
        datasource.inputs.base_directory = config['hcp_participant_data']['S1200_imaging']
        subjectlevel.inputs.modelfit.runinfo_node.data_dir = config['hcp_participant_data']['S1200_imaging']

    import pdb; pdb.set_trace()

    infosource.iterables = [('subject_id', args.subject_ids)]
    tasksource.iterables = [('task', args.tasks)]
    directionsource.iterables = [('direction', ['LR','RL'])]

    SCRATCH_DIR = args.scratch
    subjectlevel.base_dir = os.path.abspath(SCRATCH_DIR + '/workingdir')
    subjectlevel.config = {
        "execution": {
            "crashdump_dir": os.path.abspath(SCRATCH_DIR + '/crashdumps')
        }
    }

    datasink.inputs.base_directory = os.path.abspath(args.out)

    subjectlevel.inputs.whitening.inputspec.atlas = args.atlas
    subjectlevel.inputs.stdclf.inputspec.atlas = args.atlas
    subjectlevel.inputs.whclf.inputspec.atlas = args.atlas

    subjectlevel.write_graph()
    if args.n_cpus and args.n_cpus > 1:
        outgraph = subjectlevel.run(plugin='MultiProc', plugin_args={'n_procs':args.n_cpus})
    else:
        outgraph = subjectlevel.run()

