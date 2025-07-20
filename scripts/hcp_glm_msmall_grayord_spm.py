# code adapted from
# https://nipype.readthedocs.io/en/latest/users/examples/fmri_fsl.html
#
# This version redoes the HCP GLMs using SPM so that I could both get LR and 
# RL GLMs maps and implement better timeseries correction, per Diedrichsen's
# recommendation for RSA analysis. I've already implemented a version of
# this using FSL and called it hcp_glm_msmall_grayord.py. This version further
# differs by incorporating RSA analysis directly into the pipeline.
#
# our general strategy here will be to take a list of inputs (one for each run, so 
# potentially spanning directions, tasks and subjects), generate 
# "subject/session" info and contrasts for each and then merge them within 
# task (so merging direction=['LR', 'RL']) in the firstlevel workflow. We will 
# assume that contrasts are identical across directions when we do this and just 
# pick the contrast configuration for the first direction. This results in a
# list of session_infos from deignspec being submitted to level1design
# which leve1design then automaticaly processes as separate sessions in
# the same GLM (so GLM design is a block diagonal matrix). Consequently,
# there's no need for a "second level" GLM, as the first level contrasts (which
# average across corresponding beta maps) do just that.
#
# The only downside to this is that we don't get run specific t-stats, but we
# compute those post hoc using the betaToTstat interface below. We follow the
# approach described in https://www.fil.ion.ucl.ac.uk/spm/doc/books/hbf2/pdfs/Ch8.pd
# which if used with contrast codes will reproduce SPM's t-stats.

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

# this next block enables multithreading in matlab, which nipype disables by default,
# but massively increases speed of matrix math. Most of our matlab commands are also 
# called after all iterables have converged, so even if with a multithreaded nipype
# workflow this still shouldn't lead to any oversubscription.
from nipype import config, logging
cfg = dict(execution={
        'single_thread_matlab': False,
        'remove_unnecessary_outputs': False})
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
# tsnr
from geometry_vs_topography.nipype import workflows as workflows

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
from traits.api import List, Bool, Float
import os
from string import Template

# utility function for dealing with MapNode outputs
def pickfirst(files):
    if isinstance(files, list):
        return files[0]
    else:
	    return files

# this version of the RDM code interfaces with a modified version of
# +rsa.+spm.distanceLDCraw, distanceLDCrawMultTask, which takes a list
# of SPM files as input and treates them as block diagonal designs for
# a concatenated timeseries of all inputs. All design terms are
# independent. The purpose is only to make use of the existing 
# spatial whitening and distance matrix covariance estimation code.
class MultiTaskRDMInputSpec(BaseInterfaceInputSpec):
    spm_mat_files = List(File(exists=True), mandatory=True, 
        desc="List of paths to SPM.mat files produce by EstimateModel")

    atlas = File(exists=True, mandatory=False,
        desc="Path to an atlas in register with SPM betas")
        
    normmethod = traits.Enum('multivariate', 'univariate', 'none',
        usedefault=True,
        desc="spatial normalization method to use")
        
    normmode = traits.Enum('runwise', 'partwise', 'overall',
        usedefault=True,
        desc="spatial normalization mode to use")

    save_whitening_matrix = Bool(default=False, use_default=True,
        desc="Whether to save the whitening matrix. This can be quite large if you have \
        many conditions, ~n_conditions^4/2 bytes, and is stored in binary format. It is useful \
        for between subject comparisons that use individual level RDMs and pooled variance \
        estimates to compute similarity measures (e.g. WUC), but individual 'whitened' RDMs \
        can also be compared directly if you're comfortable assuming that the two RDMs \
        being compared have similar covariance structure.")

    precision = traits.Enum('single','double',
        usedefault=True,
        desc="datatype precision to use when saving whitening matrix")

class MultiTaskRDMOutputSpec(TraitedSpec):
    rdm = File(exists=True)

    whitened_rdm = File(exists=True)
    
    betanames = File(exists=True)

    whitening_matrix = File(exists=False, hash_files=False,
                            desc="Binary whitening matrix file (float32 lower triangle)")
    whitening_matrix_metadata = File(
                            exists=False, 
                            desc="Optional JSON metadata")

class MultiTaskRDM(BaseInterface):
    """
    Uses the rsatoolbox to compute crossvalidated mahalanobis distances among
    betas of interest. Optionally returns whitened rdms which
    accounts for repeated measures dependencies among distances. You will need
    spm12 and the rsatoolbox on your path, which can be obtained here:
    https://github.com/rsagroup/rsatoolbox_matlab

    Note that the atlas should be an indexed map in the same space as the one 
    in which you're running spm. So if you're running SPM on surface data that's
    been converted to nifti cubes using wb_command -cifti-convert, make sure you 
    convert your atlas the same way too.
    """

    input_spec = MultiTaskRDMInputSpec
    output_spec = MultiTaskRDMOutputSpec

    def _run_interface(self, runtime):
        import numpy as np
    
        if self.inputs.normmethod == 'multivariate':
            rdm_out = 'crossnobis_distance.csv'
            whitened_rdm_out = 'whitened_crossnobis_distance.csv'
        elif self.inputs.normmethod == 'univariate':
            rdm_out = 'standardized_distance.csv'
            whitened_rdm_out = 'whitened_standardized_distance.csv'
        else:
            rdm_out = 'euclidean_distance.csv'
            whitened_rdm_out = 'whitened_euclidean_distance.csv'

        if self.inputs.save_whitening_matrix:
            workdir = runtime.cwd
            whitening_matrix = os.path.join(workdir, 'whitening_matrix_out.bin')
            whitening_matrix_metadata = os.path.join(workdir, 'whitening_matrix_out.json')
        else:
            whitening_matrix = None
            whitening_matrix_metadata = None
    
        d = dict(atlas=self.inputs.atlas,
                 normmethod=self.inputs.normmethod,
                 normmode=self.inputs.normmode,
                 rdm_out=rdm_out,
                 whitened_rdm_out=whitened_rdm_out,
                 save_whitening_matrix=int(bool(self.inputs.save_whitening_matrix)),
                 whitening_matrix_out=whitening_matrix,
                 whitening_matrix_json=whitening_matrix_metadata,
                 precision=self.inputs.precision,
                 names_out='betanames.csv')

        # I don't know how to pass a list into the string Template, so instead
        # I save these to a txt and reimport them.
        paths = np.array(self.inputs.spm_mat_files)
        np.savetxt('spm_mat_files.csv', paths, fmt="%s", delimiter="\n")

        script = Template(
            """ spm_mat_files = textread('spm_mat_files.csv','%s\\n');
                atlas_path = '$atlas';
                normmethod = '$normmethod';
                normmode = '$normmode';
                save_whitening_matrix = $save_whitening_matrix;
                precision='$precision';


                % import atlas
                atlas_hdr = spm_vol(atlas_path);
                atlas_vols = spm_read_vols(atlas_hdr);
                [x0, y0, z0, t0] = size(atlas_vols);
                atlas = reshape(atlas_vols, x0*y0*z0, t0);

                % import data
                SPM = cell(size(spm_mat_files));
                for i = 1:length(SPM)
                    SPM{i} = importdata(spm_mat_files{i});
                    assert(length(SPM{i}.Sess) == length(SPM{1}.Sess),'Mismatched session lengths. Cannot compute RDMs');
                end

                filename = {};
                for j = 1:length(SPM)
                    for i = 1:size(SPM{j}.xY.P,1)
                        str = strsplit(SPM{j}.xY.P(i,:),','); 
                        filename{end+1} = str{1};
                    end
                end
                filename = unique(filename(:),'stable');

                vols = cell(1,length(filename));
                for i = 1:size(filename,1)
                    vols{i} = niftiread(filename{i}); 
                end
                vols = cat(4,vols{:});

                [x,y,z,t] = size(vols);

                if x ~= x0 || y ~= y0 | z ~= z0
                    error('Atlas and data dimensions are mismatched.');
                end

                Y = double(reshape(vols, x*y*z, t))';


                % set up condition vectors
                X = [];
                for i = 1:length(SPM)
                    X = blkdiag(X,SPM{i}.xX.X); 
                end

                conditions = zeros(size(X,2),1);
                col_ind = 0; % keeps track of col index from last task
                last_cond = 0;
                dof = 0;
                gSF = []; % global scaling factor
                for j = 1:length(SPM)
                    gSF = [gSF; SPM{j}.xGX.gSF];
                    for i = 1:length(SPM{j}.Sess)
                        sess_ind = SPM{j}.Sess(i).col;
                        con_ind = find(contains(SPM{j}.xX.name(sess_ind),'Task') & ~contains(SPM{j}.xX.name(sess_ind),'Cue'));
                        conditions(col_ind + sess_ind(con_ind)) = last_cond + (1:length(con_ind));
                    end
                    dof = dof + SPM{j}.xX.trRV;
                    last_cond = max(conditions);
                    col_ind = col_ind + size(SPM{j}.xX.X,2);
                end

                
                numCond = max(conditions);
                C = rsa.util.indicatorMatrix('allpairs',[1:numCond]);


                % loop over unique atlas regions and compute distance vectors
                uniq_rois = unique(atlas(:));
                uniq_rois(uniq_rois == 0) = [];
                [rdm, whitened_rdm] = deal(zeros(sum(unique(conditions)>0)*(sum(unique(conditions)>0)-1)/2, sum(unique(atlas(:))>0)));
                ncon = size(rdm,1);
                if save_whitening_matrix, fid_whitening=fopen('$whitening_matrix_out', 'w'); end
                for i = 1:length(uniq_rois)
                    this_roi = uniq_rois(i);
                    roi = any(this_roi == atlas, 2); % atlas might be overlapping searchlights across multiple volumes

                    this_Y = Y(:,roi).*gSF; % mask and apply SPM global signal scaling;
                    [d, Sig, names] = distanceLDCrawMultTask(this_Y, SPM, conditions(:), 'normmethod', normmethod, 'normmode', normmode);

                    V = (C*Sig*C').^2;
                    % V should be symmetric positive and semidefinite so we can exploit that.
                    % it's not perfect, but fairly close based on norm(d*(V^-0.5) - d/R)
                    % and much faster. Suitable for development purposes.
                    try
                        % add a tiny ridge in case numerical noise makes V only semi-definite
                        R = chol(V + 1e-10*eye(size(V)), 'upper');   % V = R'*R
                        whitened_d = d/R;
                    catch
                        % using this exclusively would take ~2 days for crossnobis, and we 
                        % compute 3x kinds of RDMs (MD, std distance, euc distance based), 
                        % so 6 days. Not viable except as a final run.
                        regV = (V + 1e-10*eye(size(V))); % regularize in case it's small
                        whitened_d = d*(regV^-0.5);
                    end

                    rdm(:,i) = d;
                    whitened_rdm(:,i) = whitened_d;
                    if save_whitening_matrix
                        tril_ind = tril(true(size(V)));
                        switch precision
                            case 'single'
                                fwrite(fid_whitening, single(V(tril_ind)), 'float32');
                            case 'double'
                                fwrite(fid_whitening, single(V(tril_ind)), 'float64');
                        end
                    end
                end

                if save_whitening_matrix, 
                    fclose(fid_whitening); 

                    switch precision
                        case 'single'
                            meta.format = 'float32';
                        case 'double'
                            meta.format = 'float64';
                    end
                    meta.shape = int32([ncon, ncon]);
                    meta.n_regions = size(rdm, 2);
                    meta.storage = 'lower_triangle';
                    meta.dof = single(dof/length(SPM{1}.Sess)); % this assumes equal session counds in all
                    meta.description = 'RDM whitening matrix collection';
                    jsonText = jsonencode(meta);
                    fid_meta = fopen('$whitening_matrix_json', 'w');
                    fwrite(fid_meta, jsonText);
                    fclose(fid_meta);
                end
                csvwrite('$rdm_out', rdm);
                csvwrite('$whitened_rdm_out', whitened_rdm);
                
                fid = fopen('$names_out','w+');
                names = names(:);
                fprintf(fid, '%s\\n', names{:});
                fclose(fid)
            """
        ).substitute(d)

        mlab = MatlabCommand(script=script, mfile=True, terminal_output='file')
        result = mlab.run()

        self.rdm_file = os.path.abspath(d['rdm_out'])
        self.whitened_rdm_file = os.path.abspath(d['whitened_rdm_out'])
        self.betanames = os.path.abspath(d['names_out'])
        if self.inputs.save_whitening_matrix:
            self.whitening_matrix = os.path.abspath(d['whitening_matrix_out'])
            self.whitening_matrix_metadata = os.path.abspath(d['whitening_matrix_json'])

        return result.runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['rdm'] = os.path.abspath(self.rdm_file)
        outputs['whitened_rdm'] = os.path.abspath(self.whitened_rdm_file)
        outputs['betanames'] = os.path.abspath(self.betanames)

        if self.inputs.save_whitening_matrix and hasattr(self, 'whitening_matrix'):
            outputs['whitening_matrix'] = self.whitening_matrix
            outputs['whitening_matrix_metadata'] = self.whitening_matrix_metadata

        return outputs
        


# this code computes a wihtin subject contrast cosine similarity metric
# by splitting data into training/test runs, computing contrasts for each
# and estimating cosine similarity between the two. Results are averaged
# across all available splits
class multiTaskWithinSimilarityInputSpec(BaseInterfaceInputSpec):
    spm_mat_files = List(File(exists=True), mandatory=True, 
        desc="List of paths to SPM.mat files produce by EstimateModel")

    atlas = File(exists=True, mandatory=False,
        desc="Path to an atlas in register with SPM betas")
        
    normmethod = traits.Enum('multivariate', 'univariate', 'none',
        usedefault=True,
        desc="spatial normalization method to use")

    shrinkage = Float(None,
        desc="If set, shrinkage is skipped and this value is used instead")

    target = traits.Enum('diagonal','scaledidentity',
        usedefault=True,
        desc = "What shrinkage target (prior) to use during covariance normalization.")

    nonlinearshrink = Bool(False,
        usedefault=True,
        desc = "Use nonlinear shrinkage for p > 50, n > 50.")

    normmode = traits.Enum('runwise', 'partwise', 'overall',
        usedefault=True,
        desc="spatial normalization mode to use")


class multiTaskWithinSimilarityOutputSpec(TraitedSpec):
    similarity = File(exists=True)

    betanames = File(exists=True)

class multiTaskWithinSimilarity(BaseInterface):
    """
    Uses the rsatoolbox to compute crossvalidated mahalanobis distances among
    betas of interest. Optionally returns whitened rdms which
    accounts for repeated measures dependencies among distances. You will need
    spm12 and the rsatoolbox on your path, which can be obtained here:
    https://github.com/rsagroup/rsatoolbox_matlab

    Note that the atlas should be an indexed map in the same space as the one 
    in which you're running spm. So if you're running SPM on surface data that's
    been converted to nifti cubes using wb_command -cifti-convert, make sure you 
    convert your atlas the same way too.
    """

    input_spec = multiTaskWithinSimilarityInputSpec
    output_spec = multiTaskWithinSimilarityOutputSpec

    def _run_interface(self, runtime):
        import numpy as np
    
        if self.inputs.normmethod == 'multivariate':
            out = 'whitened_similarity.csv'
        elif self.inputs.normmethod == 'univariate':
            out = 'standardized_similarity.csv'
        else:
            out = 'similarity.csv'

        d = dict(atlas=self.inputs.atlas,
                 normmethod=self.inputs.normmethod,
                 normmode=self.inputs.normmode
                 shrinkage=self.inputs.shrinkage,
                 target=self.inputs.target,
                 nonlinearshrink=int(bool(self.inputs.nonlinearshrink)),
                 out=out,
                 names_out='betanames.csv')

        # I don't know how to pass a list into the string Template, so instead
        # I save these to a txt and reimport them.
        paths = np.array(self.inputs.spm_mat_files)
        np.savetxt('spm_mat_files.csv', paths, fmt="%s", delimiter="\n")

        script = Template(
            """ spm_mat_files = textread('spm_mat_files.csv','%s\\n');
                atlas_path = '$atlas';
                normmethod = '$normmethod';
                normmode = '$normmode';
                shrinkage = str2double('$shrinkage');
                target = '$target';
                nonlinearshrink = $nonlinearshrink;

                if isnan(shrinkage), shrinkage = []; end
                
                % import atlas
                atlas_hdr = spm_vol(atlas_path);
                atlas_vols = spm_read_vols(atlas_hdr);
                [x0, y0, z0, t0] = size(atlas_vols);
                atlas = reshape(atlas_vols, x0*y0*z0, t0);

                % import data
                SPM = cell(size(spm_mat_files));
                for i = 1:length(SPM)
                    SPM{i} = importdata(spm_mat_files{i});
                    assert(length(SPM{i}.Sess) == length(SPM{1}.Sess),'Mismatched session lengths. Cannot compute RDMs');
                end
                
                filename = {};
                for j = 1:length(SPM)
                    for i = 1:size(SPM{j}.xY.P,1)
                        str = strsplit(SPM{j}.xY.P(i,:),','); 
                        filename{end+1} = str{1};
                    end
                end
                filename = unique(filename(:),'stable');

                vols = cell(1,length(filename));
                for i = 1:size(filename,1)
                    vols{i} = niftiread(filename{i}); 
                end
                vols = cat(4,vols{:});

                [x,y,z,t] = size(vols);

                if x ~= x0 || y ~= y0 | z ~= z0
                    error('Atlas and data dimensions are mismatched.');
                end

                Y = double(reshape(vols, x*y*z, t))';


                % set up condition vectors
                X = [];
                for i = 1:length(SPM)
                    X = blkdiag(X,SPM{i}.xX.X); 
                end

                conditions = zeros(size(X,2),1);
                col_ind = 0; % keeps track of col index from last task
                last_cond = 0;
                dof = 0;
                gSF = []; % global scaling factor
                for j = 1:length(SPM)
                    gSF = [gSF; SPM{j}.xGX.gSF];
                    for i = 1:length(SPM{j}.Sess)
                        sess_ind = SPM{j}.Sess(i).col;
                        con_ind = find(contains(SPM{j}.xX.name(sess_ind),'Task') & ~contains(SPM{j}.xX.name(sess_ind),'Cue'));
                        conditions(col_ind + sess_ind(con_ind)) = last_cond + (1:length(con_ind));
                    end
                    dof = dof + SPM{j}.xX.trRV;
                    last_cond = max(conditions);
                    col_ind = col_ind + size(SPM{j}.xX.X,2);
                end


                % loop over unique atlas regions and compute distance vectors
                uniq_rois = unique(atlas(:));
                uniq_rois(uniq_rois == 0) = [];
                similarity = zeros(sum(unique(conditions)>0), sum(unique(atlas(:))>0));
                ncon = size(similarity,1);
                fun=@(x1,x2)(x1(:)'*x2(:)/(norm(x1(:))*norm(x2(:)))); % cosine similarity metric
                for i = 1:length(uniq_rois)
                    this_roi = uniq_rois(i);
                    roi = any(this_roi == atlas, 2); % atlas might be overlapping searchlights across multiple volumes

                    this_Y = Y(:,roi).*gSF; % mask and apply SPM global signal scaling;
                    this_Y = this_Y(:,var(this_Y) > 0);
                    [similarity(:,i), names] = betweenSessionSimilarityMultiTask(this_Y, SPM, conditions(:), fun, ...
                        'normmethod', normmethod, 'normmode', normmode, ...
                        'shrinkage', shrinkage, 'target', target, 'nonlinearshrink', nonlinearshrink);
                end

                csvwrite('$out', similarity);
                
                fid = fopen('$names_out','w+');
                names = names(:);
                fprintf(fid, '%s\\n', names{:});
                fclose(fid)
            """
        ).substitute(d)

        mlab = MatlabCommand(script=script, mfile=True, single_comp_thread=False, 
            terminal_output='file')
        result = mlab.run()

        self.similarity = os.path.abspath(d['out'])
        self.betanames = os.path.abspath(d['names_out'])

        return result.runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['similarity'] = os.path.abspath(self.similarity)
        outputs['betanames'] = os.path.abspath(self.betanames)

        return outputs



# ########################## #
# Run specific configuration #
# ########################## #

import time; print("Completed imports:", time.ctime(), flush=True)

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
    (r'cifti_average_parcellated.txt',r'tsnr.csv'),
]


# ############ #
# Compute tSNR #
# ############ #

tsnrwf = workflows.init_tsnr()

# ###################### #
# Preprocessing Workflow #
# ###################### #

print("Building preproc wf:", time.ctime(), flush=True)
preproc = preproc_surf_hcp(hp_cutoff, TR)

# ########################## #
# firstlvl modeling workflow #
# ########################## #
print("Building first-level wf:", time.ctime(), flush=True)

# task specific event configuration

def runinfo(subject_id, task, direction, data_dir):
    from geometry_vs_topography.glm.designs import hcp_events
    from nipype.interfaces.base import Bunch
    from copy import deepcopy

    names, onsets, dur = hcp_events(subject_id, task, direction, data_dir)

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

# task specific contrast configuration

def select_contrasts(contrast_names):
    import re

    contrasts = []
    names = []
    for name in contrast_names:
        if bool(re.match(r'^Task-.*$',name)) and not bool(re.match(r'^Task-Cue.*$',name)):
            this_cont = [name,'T',[name],[1]]
            contrasts.append(this_cont)
            names.append(name)

    return contrasts, names

contrastselect_node = pe.Node(util.Function(input_names=['contrast_names'],
                                            output_names=['contrasts', 'names'],
                                            function=select_contrasts),
                        name='contrastselect_node')


modelfit = pe.Workflow(name='modelfit')

inputnode_modelfit = pe.Node(
    interface=util.IdentityInterface(fields=[
        'subject_id','task','direction',
        'func']),
    name='inputspec')

designspec = pe.Node(
    interface=model.SpecifySPMModel(), 
    name="designspec")

joinWithinSubject = pe.JoinNode(util.IdentityInterface(
        fields=['run_info','session_info','contrasts','contrast_names','cifti_template']),
    joinsource='directionsource',
    joinfield=['run_info','session_info','contrasts','contrast_names','cifti_template'],
    name='joinwithinsubject')

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
    
contrastestimate = pe.Node(
    interface=spm.EstimateContrast(),
    #overwrite = True,
    name = "contrastestimate"
)
contrastestimate.config = {'execution': {'remove_unnecessary_outputs': False}}

betatotstat = pe.Node(
    interface=betaToTstat(),
    name = 'betatotstat'
)

def select_betas_of_interest(beta_images, run_info):
    # this runs on single tasks assumes equal number of contrasts in "session" (i.e. direction)
    import numpy as np

    n_sess = len(run_info)
    
    #spm stacks session intercepts at the end, so we need to 
    # subract them from the count before dividing
    beta_per_sess = (len(beta_images)-n_sess)/n_sess
    
    filt_beta_images = []
    filt_beta_names = []
    for i,info in enumerate(run_info):
        ind0 = int(beta_per_sess*i)
        beta_names = set(info.conditions)
        # drop Cue condition from Motor task, since it's not of interest (trivial visual stim)
        if 'Task-Cue' in beta_names:
            beta_names.remove('Task-Cue')
        these_beta = [beta_images[ind0 + i] for i,x in enumerate(info.conditions) if x in beta_names]
        these_names = [x for i,x in enumerate(info.conditions) if x in beta_names]
        
        filt_beta_images.append(these_beta)
        filt_beta_names.append(these_names)
        
    return filt_beta_images, filt_beta_names

selectBetasOfInterest = pe.Node(util.Function(input_names=['beta_images', 'run_info'],
                                                 output_names=['beta_images', 'beta_names'],
                                                 function=select_betas_of_interest),
                                       name='selectbetasofinterest')
                                       

selectTstatsOfInterest = pe.Node(util.Function(input_names=['beta_images', 'run_info'],
                                                 output_names=['tstat_images', 'tstat_names'],
                                                 function=select_betas_of_interest),
                                       name='selecttstatsofinterest')


mergebeta = pe.MapNode(
    interface=fsl.Merge(
        dimension='t'),
    iterfield=['in_files'],
    name="mergebeta")
    
beta2cifti = pe.MapNode(
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

addbetanames = pe.MapNode(
    interface=wb_misc.SetMapNames(),
    iterfield=['in_file'],
    name="addbetanames")
    

mergeruntstats = pe.MapNode(
    interface=fsl.Merge(
        dimension='t'),
    iterfield=['in_files'],
    name="mergeruntstats")
    
runtstats2cifti = pe.MapNode(
    interface=wb_cifti.NiftiConvertCifti(reset_scalars=True),
    iterfield=['nifti_in'],
    name='runtstats2cifti')

addruntstatnames = pe.MapNode(
    interface=wb_misc.SetMapNames(),
    iterfield=['in_file'],
    name="addruntstatnames")
    
mergetL2 = pe.Node(
    interface=fsl.Merge(
        dimension='t'),
    name="mergetl2")

tL2_2cifti = pe.Node(
    interface=wb_cifti.NiftiConvertCifti(reset_scalars=True),
    name='tL2_2cifti')
    
addtl2names = pe.Node(
    interface=wb_misc.SetMapNames(),
    iterfield=['in_file'],
    name="addtl2names")
    

mergeCon = pe.Node(
    interface=fsl.Merge(
        dimension='t'),
    name="mergecon")

con2cifti = pe.Node(
    interface=wb_cifti.NiftiConvertCifti(reset_scalars=True),
    name='con2cifti')
    
addconnames = pe.Node(
    interface=wb_misc.SetMapNames(),
    iterfield=['in_file'],
    name="addconnames")
    
joinTasksMF = pe.JoinNode(util.IdentityInterface(
        fields=['tstats','con']),
    joinsource='tasksource',
    joinfield=['tstats','con'],
    name='jointasks')
    
mergeTstatsAcrossTasks = pe.Node(interface=wb_cifti.CiftiMerge(),
    name="mergetstatsacrosstasks")
    
mergeContrastsAcrossTasks = pe.Node(interface=wb_cifti.CiftiMerge(),
    name="mergecontrastsacrosstasks")

modelfit.connect([
    (inputnode_modelfit, runinfo_node, [('subject_id', 'subject_id'),
                                                 ('task','task'),
                                                 ('direction','direction')]),


    # set up SPM design
    (runinfo_node, contrastselect_node, [('contrast_names','contrast_names')]),

    (inputnode_modelfit, cifti2nifti, [('func','cifti_in')]),
    (cifti2nifti, designspec, [('out_file', 'functional_runs')]),
    (runinfo_node, designspec, [('run_info','subject_info')]),

    (designspec, joinWithinSubject, [(('session_info', pickfirst), 'session_info')]),
    (joinWithinSubject, level1design, [('session_info', 'session_info')]),


    # run SPM design
    (level1design, modelestimate, [('spm_mat_file', 'spm_mat_file')]),
    (modelestimate, vifs, [('spm_mat_file', 'spm_mat_file')]),
    (modelestimate, contrastestimate, [('spm_mat_file', 'spm_mat_file'),
                                       ('beta_images', 'beta_images'),
                                       ('residual_image', 'residual_image')]),
    (contrastselect_node, joinWithinSubject, [('contrasts', 'contrasts'),
                                              ('names', 'contrast_names')]),
    (joinWithinSubject, contrastestimate, [(('contrasts', pickfirst), 'contrasts')]),


    (inputnode_modelfit, joinWithinSubject, [(('func', pickfirst), 'cifti_template')]),


    # reassemble and merge run-level betas
    (runinfo_node, joinWithinSubject, [(('run_info', pickfirst), 'run_info')]),
    (joinWithinSubject, selectBetasOfInterest, [('run_info', 'run_info')]),
    (modelestimate, selectBetasOfInterest, [('beta_images', 'beta_images')]),
    
    (selectBetasOfInterest, mergebeta, [('beta_images', 'in_files')]),
    (selectBetasOfInterest, addbetanames, [(('beta_names', makeSetNamesList), 'map')]),
    (mergebeta, beta2cifti, [('merged_file', 'nifti_in')]),
    (joinWithinSubject, beta2cifti, [(('cifti_template', pickfirst), 'cifti_template')]),
    (beta2cifti, addbetanames, [('out_file', 'in_file')]),
    
    
    # make run-level betas into tstats
    (modelestimate, betatotstat, [('spm_mat_file', 'spm_mat_file')]),
    (betatotstat, selectTstatsOfInterest, [('tstats', 'beta_images')]),
    (joinWithinSubject, selectTstatsOfInterest, [('run_info', 'run_info')]),
    
    # merge run-level tstats
    (selectTstatsOfInterest, mergeruntstats, [('tstat_images', 'in_files')]),
    (selectTstatsOfInterest, addruntstatnames, [(('tstat_names', makeSetNamesList), 'map')]),
    (mergeruntstats, runtstats2cifti, [('merged_file', 'nifti_in')]),
    (joinWithinSubject, runtstats2cifti, [(('cifti_template', pickfirst), 'cifti_template')]),
    (runtstats2cifti, addruntstatnames, [('out_file', 'in_file')]),
    
    
    # merge contrasts (task-level betas)
    (contrastestimate, mergeCon, [('con_images', 'in_files')]),
    (mergeCon, con2cifti, [('merged_file', 'nifti_in')]),
    (joinWithinSubject, con2cifti, [(('cifti_template', pickfirst), 'cifti_template')]),
    
    (con2cifti, addconnames, [('out_file', 'in_file')]),
    (joinWithinSubject, addconnames, [(('contrast_names', makeSetNamesList), 'map')]),
    
    (addconnames, joinTasksMF, [('out_file', 'con')]),
    (joinTasksMF, mergeContrastsAcrossTasks, [('con', 'cifti')]),
    

    # merge task-level tstats
    (contrastestimate, mergetL2, [('spmT_images', 'in_files')]),
    (mergetL2, tL2_2cifti, [('merged_file', 'nifti_in')]),
    (joinWithinSubject, tL2_2cifti, [(('cifti_template', pickfirst), 'cifti_template')]),
    
    (tL2_2cifti, addtl2names, [('out_file', 'in_file')]),
    (joinWithinSubject, addtl2names, [(('contrast_names', makeSetNamesList), 'map')]),
    
    (addtl2names, joinTasksMF, [('out_file', 'tstats')]),
    (joinTasksMF, mergeTstatsAcrossTasks, [('tstats', 'cifti')]),    
])

################################
# do spatial whitening and RSA #
################################
#
# Note, we don't need to combine these in the same workflow, but it makes sense to do so since
# the whitened betas are the things that RSA operates on internally. The RSA function doesn't 
# output the whitened betas but we can replicate the process externally and save them for 
# subsequent evaluation.
print("Building RSA wf:", time.ctime(), flush=True)

rsawf = pe.Workflow(name='rsa')

inputnode_rsa = pe.Node(
    interface=util.IdentityInterface(fields=[
        'spm_mat_file','atlas', 'run_info']),
    name='inputspec')

atlas2nifti = pe.Node(
    interface=wb_cifti.CiftiConvertNifti(
        smaller_dims=True),
    iterfield=['cifti_in'],
    name='atlas2nifti')
        
joinTaskSPMMats = pe.JoinNode(util.IdentityInterface(
        fields=['spm_mat_file']),
    joinsource='tasksource',
    joinfield=['spm_mat_file'],
    name='jointaskspmmats')
    
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
                                       
mergestdbetas = pe.MapNode(
    interface=fsl.Merge(
        dimension='t'),
    iterfield=['in_files'],
    name="mergestdbetas")
    
standardizedbeta2cifti = pe.MapNode(
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

addstdnames = pe.MapNode(
    interface=wb_misc.SetMapNames(),
    iterfield=['in_file'],
    name="addstdnames")
    

# spatial standardize overall
stdbetasl2 = pe.Node(
    interface=SpatialWhitening(
        normmode='overall', 
        shrinkage=1.0),
    name="stdbetasl2")
    
splitstdbetasl2 = pe.Node(
    interface=fsl.Split(dimension='t'),
    name="splitstdbetasl2")

selectStdBetasOfInterestl2 = pe.Node(util.Function(input_names=['beta_images', 'run_info'],
                                                 output_names=['beta_images', 'beta_names'],
                                                 function=select_betas_of_interest),
                                       name='selectstdbetasofinterestl2')
                                       
mergestdbetasl2 = pe.MapNode(
    interface=fsl.Merge(
        dimension='t'),
    iterfield=['in_files'],
    name="mergestdbetasl2")
    
standardizedbeta2ciftil2 = pe.MapNode(
    interface=wb_cifti.NiftiConvertCifti(reset_scalars=True),
    iterfield=['nifti_in'],
    name='standardizedbeta2ciftil2')

addstdnamesl2 = pe.MapNode(
    interface=wb_misc.SetMapNames(),
    iterfield=['in_file'],
    name="addstdnamesl2")

estStdContrasts = pe.Node(
    interface=wb_cifti.Average(),
    name="eststdcontrasts")
    
    
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
                                       
mergewhitenedbetas = pe.MapNode(
    interface=fsl.Merge(
        dimension='t'),
    iterfield=['in_files'],
    name="mergewhitenedbetas")
    
whitenedbeta2cifti = pe.MapNode(
    interface=wb_cifti.NiftiConvertCifti(reset_scalars=True),
    iterfield=['nifti_in'],
    name='whitenedbeta2cifti')

addwhitenednames = pe.MapNode(
    interface=wb_misc.SetMapNames(),
    iterfield=['in_file'],
    name="addwhitenednames")
        

# spatial whitening overall
whitenbetasl2 = pe.Node(
    interface=SpatialWhitening(
        normmode='overall'),
    name="whitenbetasl2")
    
splitwhitenedbetasl2 = pe.Node(
    interface=fsl.Split(dimension='t'),
    name="splitwhitenedbetasl2")

selectWhitenedBetasOfInterestl2 = pe.Node(util.Function(input_names=['beta_images', 'run_info'],
                                                 output_names=['beta_images', 'beta_names'],
                                                 function=select_betas_of_interest),
                                       name='selectwhitenedbetasofinterestl2')
                                       
mergewhitenedbetasl2 = pe.MapNode(
    interface=fsl.Merge(
        dimension='t'),
    iterfield=['in_files'],
    name="mergewhitenedbetasl2")
    
whitenedbeta2ciftil2 = pe.MapNode(
    interface=wb_cifti.NiftiConvertCifti(reset_scalars=True),
    iterfield=['nifti_in'],
    name='whitenedbeta2ciftil2')

addwhitenednamesl2 = pe.MapNode(
    interface=wb_misc.SetMapNames(),
    iterfield=['in_file'],
    name="addwhitenednamesl2")
    
estWhitenedContrasts = pe.Node(
    interface=wb_cifti.Average(),
    name="estwhitenedcontrasts")    
    
    
    
# merge whitened betas for post-hoc between subject spatial correlation analysis

mergeWhitenedContrastsAcrossTasks = pe.Node(interface=wb_cifti.CiftiMerge(),
    name="mergewhitenedcontrastsacrosstasks")
    

mergeStdContrastsAcrossTasks = pe.Node(interface=wb_cifti.CiftiMerge(),
    name="mergestandardizedcontrastsacrosstasks")
    

# within subject cosine similarity
stdCosim = pe.Node(
    interface=multiTaskWithinSimilarity(
        normmethod='univariate'),
    name='stdcosim')

whitenedCosim = pe.Node(
    interface=multiTaskWithinSimilarity(
        normmethod='multivariate'),
    name='whitenedcosim')
    

# RSA
eucdist = pe.Node(
    interface=MultiTaskRDM(normmethod='none'),
    name="eucdist")
    
stddist = pe.Node(
    interface=MultiTaskRDM(
        normmethod='univariate',
        save_whitening_matrix=True),
    name="stddist")
    
crossnobis = pe.Node(
    interface=MultiTaskRDM(
        normmethod='multivariate',
        save_whitening_matrix=True),
    name="crossnobis")

rsawf.connect([
    (inputnode_rsa, atlas2nifti, [('atlas', 'cifti_in')]),
    
    
    # standardize runwise
    (inputnode_rsa, stdbetas, [('spm_mat_file', 'spm_mat_file')]),
    (atlas2nifti, stdbetas, [('out_file', 'atlas')]),
    
    # split std betas by session, merge and convert to LR and RL specific ciftis
    (stdbetas, splitstdbetas, [('whitened_images', 'in_file')]),
    (splitstdbetas, selectStdBetasOfInterest, [
        ('out_files', 'beta_images')]),
    (inputnode_rsa, selectStdBetasOfInterest, [
        ('run_info', 'run_info')]),
        
    (selectStdBetasOfInterest, mergestdbetas, [('beta_images', 'in_files')]),
    (mergestdbetas, standardizedbeta2cifti, [('merged_file', 'nifti_in')]),
    (inputnode_rsa, standardizedbeta2cifti, [('atlas', 'cifti_template')]),
    
    # assign condition names to whitened betas
    (selectStdBetasOfInterest, addstdnames, [(('beta_names', makeSetNamesListSubjLevel), 'map')]),
    (standardizedbeta2cifti, addstdnames, [('out_file', 'in_file')]),

    
    # standardize subject-wise ("overall" in rsatoolbox'select_contrasts parlance)
    (inputnode_rsa, stdbetasl2, [('spm_mat_file', 'spm_mat_file')]),
    (atlas2nifti, stdbetasl2, [('out_file', 'atlas')]),

    # split std betas by session, merge and convert to overall average specific ciftis
    (stdbetasl2, splitstdbetasl2, [('whitened_images', 'in_file')]),
    (splitstdbetasl2, selectStdBetasOfInterestl2, [
        ('out_files', 'beta_images')]),
    (inputnode_rsa, selectStdBetasOfInterestl2, [
        ('run_info', 'run_info')]),
        
    (selectStdBetasOfInterestl2, mergestdbetasl2, [('beta_images', 'in_files')]),
    (mergestdbetasl2, standardizedbeta2ciftil2, [('merged_file', 'nifti_in')]),
    (inputnode_rsa, standardizedbeta2ciftil2, [('atlas', 'cifti_template')]),
    
    # assign condition names to whitened betas
    (selectStdBetasOfInterestl2, addstdnamesl2, [(('beta_names', makeSetNamesListSubjLevel), 'map')]),
    (standardizedbeta2ciftil2, addstdnamesl2, [('out_file', 'in_file')]),

    # this produces files equivalent to a standardized con_XXXX.nii file (more or less,
    # they are standardized by within session noise covariance after all)
    (addstdnamesl2, estStdContrasts, [('out_file', 'in_vars')]),
    
    (estStdContrasts, joinTaskBetas, [('out_file', 'standardized_betas')]),
    (joinTaskBetas, mergeStdContrastsAcrossTasks, [('standardized_betas', 'cifti')]),
    
    
    # whitten runwise
    (inputnode_rsa, whitenbetas, [('spm_mat_file', 'spm_mat_file')]),
    (atlas2nifti, whitenbetas, [('out_file', 'atlas')]),
    
    # split whitened betas by session, merge and convert to LR and RL specific ciftis
    (whitenbetas, splitwhitenedbetas, [('whitened_images', 'in_file')]),
    (splitwhitenedbetas, selectWhitenedBetasOfInterest, [
        ('out_files', 'beta_images')]),
    (inputnode_rsa, selectWhitenedBetasOfInterest, [
        ('run_info', 'run_info')]),
        
    (selectWhitenedBetasOfInterest, mergewhitenedbetas, [('beta_images', 'in_files')]),
    (mergewhitenedbetas, whitenedbeta2cifti, [('merged_file', 'nifti_in')]),
    (inputnode_rsa, whitenedbeta2cifti, [('atlas', 'cifti_template')]),
    
    # assign condition names to whitened betas
    (selectWhitenedBetasOfInterest, addwhitenednames, [(('beta_names', makeSetNamesListSubjLevel), 'map')]),
    (whitenedbeta2cifti, addwhitenednames, [('out_file', 'in_file')]),

    # whiten overall-wise
    (inputnode_rsa, whitenbetasl2, [('spm_mat_file', 'spm_mat_file')]),
    (atlas2nifti, whitenbetasl2, [('out_file', 'atlas')]),
    
    # split whitened betas by session, merge and convert to session specific ciftis
    (whitenbetasl2, splitwhitenedbetasl2, [('whitened_images', 'in_file')]),
    (splitwhitenedbetasl2, selectWhitenedBetasOfInterestl2, [
        ('out_files', 'beta_images')]),
    (inputnode_rsa, selectWhitenedBetasOfInterestl2, [
        ('run_info', 'run_info')]),
        
    (selectWhitenedBetasOfInterestl2, mergewhitenedbetasl2, [('beta_images', 'in_files')]),
    (mergewhitenedbetasl2, whitenedbeta2ciftil2, [('merged_file', 'nifti_in')]),
    (inputnode_rsa, whitenedbeta2ciftil2, [('atlas', 'cifti_template')]),
    
    # assign condition names to whitened betas
    (selectWhitenedBetasOfInterestl2, addwhitenednamesl2, [(('beta_names', makeSetNamesListSubjLevel), 'map')]),
    (whitenedbeta2ciftil2, addwhitenednamesl2, [('out_file', 'in_file')]),

    # this produces files equivalent to a whitened con_XXXX.nii file (more or less,
    # they are standardized by within session noise covariance after all)
    (addwhitenednamesl2, estWhitenedContrasts, [('out_file', 'in_vars')]),
    
    (estWhitenedContrasts, joinTaskBetas, [('out_file', 'whitened_betas')]),
    (joinTaskBetas, mergeWhitenedContrastsAcrossTasks, [('whitened_betas', 'cifti')]),
    

    # estimate within subject cosine similarity
    (inputnode_rsa, joinTaskSPMMats, [('spm_mat_file', 'spm_mat_file')]),

    (joinTaskSPMMats, whitenedCosim, [('spm_mat_file', 'spm_mat_files')]),
    (atlas2nifti, whitenedCosim, [(('out_file', pickfirst), 'atlas')]),
    
    (joinTaskSPMMats, stdCosim, [('spm_mat_file', 'spm_mat_files')]),
    (atlas2nifti, stdCosim, [(('out_file', pickfirst), 'atlas')]),

    
    # do RSAs    
    (joinTaskSPMMats, crossnobis, [('spm_mat_file', 'spm_mat_files')]),
    (atlas2nifti, crossnobis, [(('out_file', pickfirst), 'atlas')]),
    
    (joinTaskSPMMats, eucdist, [('spm_mat_file', 'spm_mat_files')]),
    (atlas2nifti, eucdist, [(('out_file', pickfirst), 'atlas')]),
    
    (joinTaskSPMMats, stddist, [('spm_mat_file', 'spm_mat_files')]),
    (atlas2nifti, stddist, [(('out_file', pickfirst), 'atlas')]),
    
])

################################################################
# Combine preproc, first level and spatial whitening workflows #
################################################################
print("Building subject-level workflow:", time.ctime(), flush=True)

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
        
    (datasource, tsnrwf, [('func_surf', 'inputspec.in_files')]),

    (infosource, modelfit, [('subject_id', 'inputspec.subject_id')]),
    (tasksource, modelfit, [('task','inputspec.task')]),
    (directionsource, modelfit, [('direction','inputspec.direction')]),
    (preproc, modelfit, [('nii2cifti.out_file', 'inputspec.func')]),
        
    (modelfit, rsawf, [('contrastestimate.spm_mat_file', 'inputspec.spm_mat_file'),
                         ('joinwithinsubject.run_info', 'inputspec.run_info'),
                      ]),
    
    # save desired outputs
    # datasink is being run on both LR and RL iterations, even though outputs should be merged for most of these. Why?
    # maybe it's the l1_betas?
    (tsnrwf, datasink, [('outputspec.out_file', 'results.@tsnr')]),


    (modelfit, datasink, [('addbetanames.out_file', 'results.@l1_betas'),
                          ('addruntstatnames.out_file', 'results.@l1_tstats'),
    
                          #('addtl2names.out_file', 'results.@l2_tstats'), # We output these merged across tasks
                          ('mergecontrastsacrosstasks.out_file', 'results.all_tasks.contrasts'),
    
                          #('addconnames.out_file',  'results.@l2_cons'), # we output these merged across tasks
                          ('mergetstatsacrosstasks.out_file', 'results.all_tasks.tstats'),
                          
                          ('vifs.vifs', 'results.@vifs'),
                          ('vifs.png', 'results.@png'),
                          ('vifs.hpfilt', 'results.@hpfilt'),
                          
                          ('contrastestimate.spm_mat_file', 'results.@spm_mat_file')
                          ]),
                          
    
    (rsawf, datasink, [('crossnobis.rdm', 'results.all_tasks.rsa.crossnobis.@rdm'),
                       ('crossnobis.whitened_rdm', 'results.all_tasks.rsa.crossnobis.@whitened_rdm'),
                       ('crossnobis.betanames', 'results.all_tasks.rsa.crossnobis.@betanames'),
                       ('crossnobis.whitening_matrix', 'results.all_tasks.rsa.crossnobis.@whitening_matrix'),
                       ('crossnobis.whitening_matrix_metadata', 'results.all_tasks.rsa.crossnobis.@whitening_matrix_metadata'),
                       
                       ('eucdist.rdm', 'results.all_tasks.rsa.eucdist.@rdm'),
                       ('eucdist.whitened_rdm', 'results.all_tasks.rsa.eucdist.@whitened_rdm'),
                       ('eucdist.betanames', 'results.all_tasks.rsa.eucdist.@betanames'),
                       
                       ('stddist.rdm', 'results.all_tasks.rsa.stddist.@rdm'),
                       ('stddist.whitened_rdm', 'results.all_tasks.rsa.stddist.@whitened_rdm'),
                       ('stddist.betanames', 'results.all_tasks.rsa.stddist.@betanames'),
                       ('stddist.whitening_matrix', 'results.all_tasks.rsa.stddist.@whitening_matrix'),
                       ('stddist.whitening_matrix_metadata', 'results.all_tasks.rsa.stddist.@whitening_matrix_metadata'),
    
                        
                       ('whitenedcosim.similarity', 'results.all_tasks.whitened_contrasts.@cosim'),
                       ('whitenedcosim.betanames', 'results.all_tasks.whitened_contrasts.@betanames'),
    
                       ('stdcosim.similarity', 'results.all_tasks.standardized_contrasts.@cosim'),
                       ('stdcosim.betanames', 'results.all_tasks.standardized_contrasts.@betanames'),
    
    
                       ('addstdnames.out_file', 'results.@l1_standardized_betas'), # run specific (LR/RL) task tstats
                       #(('eststdcontrasts.out_file', pickfirst), 'results.standardized_betas.@l2_std_betas'), # subject level task stats. We output these merged acros tasks
                       ('mergestandardizedcontrastsacrosstasks.out_file', 
                        'results.all_tasks.standardized_contrasts'), # these are averaged across run-level standardized betas
    
                       ('addwhitenednames.out_file', 'results.@l1_whitened_betas'), # task x run specific tstats
                       #(('estwhitenedcontrasts.out_file', pickfirst), 'results.whitened_betas.@l2_whitened_betas'), # subject level task stats. We output these merged across tasks.
                       ('mergewhitenedcontrastsacrosstasks.out_file', 'results.all_tasks.whitened_contrasts'), # these are averaged across run-level whitened betas
                       #('mergetstatsacrosstasks.out_file', 'results.all_subjectlevel_tstats')
                       ])
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
    print("Executing main():", time.ctime(), flush=True)

    parser = argparse.ArgumentParser(description="HCP single trials GLM estimation")
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
                        help='cifti dlabels file to use to define RSA parcells')
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

    print(f'Using {datasource.inputs.base_directory} as input directory')

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
    
    subjectlevel.inputs.tsnr.inputspec.atlas = args.atlas
    subjectlevel.inputs.rsa.inputspec.atlas = args.atlas

    print("Starting subject level workflow:", time.ctime(), flush=True)
    subjectlevel.write_graph()

    if args.n_cpus and args.n_cpus > 1:
        outgraph = subjectlevel.run(plugin='MultiProc', plugin_args={'n_procs':args.n_cpus})
    else:
        outgraph = subjectlevel.run()

