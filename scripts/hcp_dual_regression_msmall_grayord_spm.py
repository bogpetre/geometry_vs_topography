# code adapted from
# https://nipype.readthedocs.io/en/latest/users/examples/fmri_fsl.html
#
# This script performs dual regression in the surface HCP data and then computes
# whitened crossnobis distances between components for each of the CANlab2024 atlas
# ROIs. It is a hybrid of the methods used to produce the PTN1200 data and the 
# methods described by glasser et al. In particular it differs from the PTN1200 
# release in the use of 
# - weighted spatial regression
# - iterative dual regression
# - confound correction
# Meanwhile it differs from the Glasser approach in the use of
# - quasi-hierarchical dual regression (rather than timeseries concatenation),
# - absence of unstructured noise variance normalization
# - More iterations of dual regression (5 vs. 2)
# Qualitatively it seems like the resultant networks show better contrast than
# subject specific networks of the PTN release. Replicating the PTN methods on
# each session separately and comparing the results to this implementation also
# shows better test-retest reliability for this method
#
# Preprocessing:
# - 24 motion vector, csf and white matter signal regression
#
# Bogdan Petre
# Oct 30, 2024

# ToDo: There's a bit left to do
# - Make dual regression iteration count a commandline arguments
#
# - See about incorporating confounds into dual regression algorithm
#
# - Consider implementing group ICA for pairs of subjects.
#   There may be no need for dual regression if we can get the component timeseries
#   directly from the the group ICA, but think more about why dual regression starts 
#   with maps to regenerate timeseries rather than simply using these directly. There
#   may be a catch.

from __future__ import print_function
from __future__ import division
from builtins import str
from builtins import range

import os
import argparse
import numpy as np

import nipype.interfaces.io as nio  # Data i/o
import nipype.interfaces.fsl as fsl  # neuroimaging library with useful utilities
import nipype.interfaces.spm as spm  # neuroimaging library we use for GLM
import nipype.pipeline.engine as pe  # pypeline engine
import nipype.interfaces.utility as util  # utility
import nipype.algorithms.modelgen as model  # model generation
import nipype.algorithms.rapidart as ra

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
mlab.MatlabCommand.set_default_paths(['/dartfs-hpc/rc/home/m/f0042vm/software/spm12',
                                      '/dartfs-hpc/rc/home/m/f0042vm/software/rsatoolbox_matlab',
                                      '/dartfs-hpc/rc/lab/C/CANlab/labdata/projects/bogdan_hcp_glm/libraries/matlab'])

# Without this hack this script tends to hang when run over SLURM on NSF filesystems
from nipype.interfaces.spm import SPMCommand
SPMCommand.version = "12.7777"  # any dummy version string
v = SPMCommand().version

from nipype_workbench_ext import cifti as wb_cifti

from geometry_vs_topography.nipype import workflows as aligntools_wf # for dual regression workflow

# the following libraries are needed for confound correction
from nipype.interfaces.freesurfer import Binarize
#import nipype.algorithms.rapidart as ra # idiosyncratic spikes/run break unbiased distances

import sys

package_directory = '/dartfs-hpc/rc/lab/C/CANlab/labdata/projects/bogdan_hcp_glm/libraries/'
if package_directory not in sys.path:
    sys.path.insert(0, package_directory)
    
# an interface to the rsatoolbox_matlab repo's spatial whitening tools
from glm.rsa import SpatialWhitening, WithinSimilarity

package_directory = '/dartfs-hpc/rc/home/m/f0042vm/software/canlab/CanlabCore/nipype/'
if package_directory not in sys.path:
    sys.path.insert(0, package_directory)
import canlabCore.preproc as canlabCore

fsl.FSLCommand.set_default_output_type('NIFTI_GZ')

data_dir = os.path.abspath('/dartfs/rc/lab/D/DBIC/DBIC/archive/HCP/HCP1200')

# cutoff in seconds
hp_cutoff=200 # HCP default, already applied to *hp2000* data
TR = 0.72

#####################
# Custom Interfaces #
#####################
from nipype.interfaces.matlab import MatlabCommand
from nipype.interfaces.base import (
    TraitedSpec,
    BaseInterface,
    BaseInterfaceInputSpec,
    File,
)
from nipype.utils.filemanip import save_json
import nipype.interfaces.fsl as fsl # for FSLCommand and its input/out specs
from traits.api import List, Bool, Float, Enum, Any
import os
from string import Template


class RDMInputSpec(BaseInterfaceInputSpec):
    spm_mat_file = File(exists=True, mandatory=True, 
        desc="Paths to SPM.mat file produce by EstimateModel")

    atlas = File(exists=True, mandatory=False,
        desc="Path to an atlas in register with SPM betas")
        
    normmethod = Enum('multivariate', 'univariate', 'none',
        usedefault=True,
        desc="spatial normalization method to use")
        
    normmode = Enum('runwise', 'overall',
        usedefault=True,
        desc="spatial normalization mode to use")

    shrinkage = Float(None,
        desc="If set, shrinkage is skipped and this value is used instead")

    target = Enum('diagonal','scaledidentity',
        usedefault=True,
        desc = "What shrinkage target (prior) to use during covariance normalization.")

    nonlinearshrink = Bool(False,
        usedefault=True,
        desc = "Use nonlinear shrinkage for p > 50, n > 50.")
    
    save_whitening_matrix = Bool(default=False, use_default=True,
        desc="Whether to save the whitening matrix. This can be quite large if you have \
        many conditions, ~n_conditions^4/2 bytes, and is stored in binary format. It is useful \
        for between subject comparisons that use individual level RDMs and pooled variance \
        estimates to compute similarity measures (e.g. WUC), but individual 'whitened' RDMs \
        can also be compared directly if you're comfortable assuming that the two RDMs \
        being compared have similar covariance structure.")

class RDMOutputSpec(TraitedSpec):
    rdm = File(exists=True)

    whitened_rdm = File(exists=True)
    
    betanames = File(exists=True)

    whitening_matrix = File(exists=False, hash_files=False,
                            desc="Binary whitening matrix file (float32 lower triangle)")
    whitening_matrix_metadata = File(
                            exists=False, 
                            desc="Optional JSON metadata")


class RDM(BaseInterface):
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

    input_spec = RDMInputSpec
    output_spec = RDMOutputSpec

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
                 shrinkage=self.inputs.shrinkage,
                 target=self.inputs.target,
                 nonlinearshrink=int(bool(self.inputs.nonlinearshrink)),
                 spm_mat_file=self.inputs.spm_mat_file,
                 rdm_out=rdm_out,
                 whitened_rdm_out=whitened_rdm_out,
                 save_whitening_matrix=int(bool(self.inputs.save_whitening_matrix)),
                 whitening_matrix_out=whitening_matrix,
                 whitening_matrix_json=whitening_matrix_metadata,
                 names_out='betanames.csv')

        script = Template(
            """ spm_mat_file = '$spm_mat_file';
                atlas_path = '$atlas';
                normmethod = '$normmethod';
                normmode = '$normmode';
                shrinkage = str2double('$shrinkage');
                target = '$target';
                nonlinearshrink = $nonlinearshrink;
                save_whitening_matrix = $save_whitening_matrix;

                if isnan(shrinkage), shrinkage = []; end
                
                % import data
                SPM = importdata(spm_mat_file);

                filename = {};
                for i = 1:size(SPM.xY.P,1)
                    str = strsplit(SPM.xY.P(i,:),','); 
                    filename{end+1} = str{1};
                end
                filename = unique(filename(:),'stable');

                vols = cell(1,length(filename));
                for i = 1:size(filename,1)
                    vols{i} = niftiread(filename{i}); 
                end
                vols = cat(4,vols{:});

                [x,y,z,t] = size(vols);

                Y = double(reshape(vols, x*y*z, t))';

                % import atlas
                if ~isempty(atlas_path) && ~strcmp(atlas_path,'<undefined>')
                    atlas_hdr = spm_vol(atlas_path);
                    atlas_vols = spm_read_vols(atlas_hdr);
                    [x0, y0, z0, t0] = size(atlas_vols);
                    atlas = reshape(atlas_vols, x0*y0*z0, t0);

                    if x ~= x0 || y ~= y0 | z ~= z0
                        error('Atlas and data dimensions are mismatched.');
                    end
                else
                    atlas = zeros(x*y*z,1);
                    atlas(var(Y) > eps) = 1;
                end

                % set up condition vectors
                X = SPM.xX.X;

                conditions = zeros(size(X,2),1);
                gSF = SPM.xGX.gSF;
                for i = 1:length(SPM.Sess)
                    sess_ind = SPM.Sess(i).col;
                    con_ind = find(contains(SPM.xX.name(sess_ind),{'Task-','ICA'}) & ~contains(SPM.xX.name(sess_ind),'Cue'));
                    conditions(sess_ind(con_ind)) = 1:length(con_ind);
                end

                %{
                numReg = size(X,2);
                conditionVec = conditions;
                numCond = max(conditionVec);
                if (length(conditionVec)<numReg)
                    conditionVec=[conditionVec;zeros(numReg-length(conditionVec),1)];
                end
                Z = rsa.util.indicatorMatrix('identity_p',conditionVec);
                nonInterest = all(Z==0,2);   % Regressors not in the conditions
                numNonInterest = sum(nonInterest);
                Z(nonInterest,end+1:end+sum(numNonInterest))=eye(numNonInterest);
                C = rsa.util.indicatorMatrix('allpairs',[1:numCond]);
                %}
                
                numCond = max(conditions);
                C = rsa.util.indicatorMatrix('allpairs',[1:numCond]);


                % loop over unique atlas regions and compute distance vectors
                uniq_rois = unique(atlas(:));
                uniq_rois(uniq_rois == 0) = [];
                [rdm, whitened_rdm] = deal(nan(sum(unique(conditions)>0)*(sum(unique(conditions)>0)-1)/2, sum(unique(atlas(:))>0)));
                ncon = size(rdm,1);
                if save_whitening_matrix, fid_whitening=fopen('$whitening_matrix_out', 'w'); end
                for i = 1:length(uniq_rois)
                    this_roi = uniq_rois(i);
                    roi = any(this_roi == atlas, 2); % atlas might be overlapping searchlights across multiple volumes

                    this_Y = Y(:,roi).*gSF; % mask and apply SPM global signal scaling;
                    this_Y = this_Y(:,var(this_Y) > 0);
                    if isempty(this_Y)
                        if save_whitening_matrix
                            nCond = length(unique(conditions)) - 1; % subtract 1 to account for condition=0
                            nCovElem = nCond*(nCond - 1)/2;
                            V = eye(nCovElem);

                            tril_ind = tril(true(size(V)));
                            fwrite(fid_whitening, single(V(tril_ind)), 'float32');
                        end
                        continue
                    end
                    [d, Sig, names] = rsa.spm.distanceLDCraw(this_Y, SPM, conditions(:), 'normmethod', normmethod, 'normmode', normmode, ...
                        'shrinkage', shrinkage, 'target', target,'nonlinearshrink',nonlinearshrink);

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
                        fwrite(fid_whitening, single(V(tril_ind)), 'float32');
                    end
                end

                if save_whitening_matrix, 
                    fclose(fid_whitening); 

                    meta.format = 'float32';
                    meta.shape = int32([ncon, ncon]);
                    meta.n_regions = size(rdm, 2);
                    meta.storage = 'lower_triangle';
                    meta.dof = single(SPM.xX.trRV/length(SPM.Sess));
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

        mlab = MatlabCommand(script=script, mfile=True, single_comp_thread=False, 
            terminal_output='file')
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
        
        

class NVolsInputSpec(fsl.base.FSLCommandInputSpec):
    in_file = File(
        exists=True,
        argstr="%s",
        mandatory=True,
        position=1,
        desc="input file to generate stats of",
    )


class NVolsOutputSpec(TraitedSpec):
    nvols = Any(desc="nvols output")
            
class NVols(fsl.base.FSLCommand):
    """Use FSL fslnvol command to read out 4d vol count. Based on nipype FSL.ImageStats
    `FSL info
    <http://www.fmrib.ox.ac.uk/fslcourse/lectures/practicals/intro/index.htm#fslutils>`_


    Examples
    --------

    >>> from nipype.interfaces.fsl import NVol
    >>> from nipype.testing import funcfile
    >>> stats = NVols(in_file=funcfile)
    >>> stats.cmdline == 'fslvols %s'%funcfile
    True


    """

    input_spec = NVolsInputSpec
    output_spec = NVolsOutputSpec

    _cmd = "fslnvols"

    def _format_arg(self, name, trait_spec, value):
        return super()._format_arg(name, trait_spec, value)

    def aggregate_outputs(self, runtime=None, needed_outputs=None):
        outputs = self._outputs()
        # local caching for backward compatibility
        outfile = os.path.join(os.getcwd(), "nvols_result.json")
        if runtime is None:
            try:
                nvols = load_json(outfile)["nvols"]
            except OSError:
                return self.run().outputs
        else:
            nvols = int(runtime.stdout)
            save_json(outfile, dict(nvols=nvols))
        outputs.nvols = nvols
        return outputs
        
        
        
class FMRIConcatInputSpec(BaseInterfaceInputSpec):
    spm_mat_file = File(exists=True, mandatory=True, 
        desc="path to SPM file in main SPM directory (containing the betas)")
        
    nscan = List(
        mandatory=True,
        desc="(list (n,) indicating number of TRs of each constituent scan.")

class FMRIConcatOutputSpec(TraitedSpec):
    spm_mat_file = File(exists=True)


class FMRIConcat(BaseInterface):
    """
    Uses modified spm_fmri_concatenate function to adjust temporal 
    autocorrelation function for concatenation of scans
    """

    input_spec = FMRIConcatInputSpec
    output_spec = FMRIConcatOutputSpec

    def _run_interface(self, runtime):
        import shutil
        import os
        
        basename = os.path.basename(self.inputs.spm_mat_file)
        newSPM = os.path.join(os.getcwd(), basename)
        shutil.copyfile(self.inputs.spm_mat_file, newSPM)
        
        d = dict(spm_mat_file=newSPM,
                 nscan=[int(n) for n in self.inputs.nscan])

        # This is your MATLAB code template
        script = Template(
            """ spm_mat_file = '$spm_mat_file';
                nscan = $nscan;
                spm_fmri_concatenate_multisess(spm_mat_file, nscan);
            """
        ).substitute(d)

        # mfile = True  will create an .m file with your script and executed.
        # Alternatively
        # mfile can be set to False which will cause the matlab code to be
        # passed
        # as a commandline argument to the matlab executable
        # (without creating any files).
        # This, however, is less reliable and harder to debug
        # (code will be reduced to
        # a single line and stripped of any comments).
        mlab = MatlabCommand(script=script, mfile=True)
        result = mlab.run()

        self.spm_mat_file = os.path.abspath(d['spm_mat_file'])
        
        base, ext = os.path.splitext(basename)
        os.remove(os.path.join(os.getcwd(), base + '_backup' + ext))

        return result.runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['spm_mat_file'] = os.path.abspath(self.spm_mat_file)
        return outputs

# ########################## #
# Run specific configuration #
# ########################## #

subjectsource = pe.Node(
    interface=util.IdentityInterface(fields=['subject_id']), name="subjectsource")

sessionsource = pe.Node(
    interface=util.IdentityInterface(fields=['session']), name="sessionsource")

directionsource = pe.Node(
    interface=util.IdentityInterface(fields=['direction']), name="directionsource")

datasourcefunc = pe.Node(
    interface=nio.DataGrabber(
        infields=['subject_id', 'session', 'direction'], 
        outfields=['func', 'vol', 'motion']),
    name='datasourcefunc')
datasourcefunc.inputs.base_directory = data_dir
datasourcefunc.inputs.template='*'
datasourcefunc.inputs.field_template={'func': '%s/MNINonLinear/Results/rfMRI_REST%s_%s/rfMRI_REST%s_%s_Atlas_MSMAll_hp2000_clean.dtseries.nii',
                            'vol': '%s/MNINonLinear/Results/rfMRI_REST%s_%s/rfMRI_REST%s_%s_hp2000_clean.nii.gz',
                            'motion': '%s/MNINonLinear/Results/rfMRI_REST%s_%s/Movement_Regressors_dt.txt',}
datasourcefunc.inputs.template_args = {'func': [['subject_id','session','direction','session','direction']],
                                   'vol': [['subject_id','session','direction','session','direction']],
                                   'motion': [['subject_id','session','direction']]}
datasourcefunc.inputs.sort_filelist = True


datasourceanat = pe.Node(
    interface=nio.DataGrabber(
        infields=['subject_id'], 
        outfields=['surface_left', 'surface_right', 'roi_left', 'roi_right', 'seg']),
    name='datasourceanat')
datasourceanat.inputs.base_directory = data_dir
datasourceanat.inputs.template='*'
datasourceanat.inputs.field_template={'seg': '%s/MNINonLinear/aparc+aseg.nii.gz',
                            'surface_left': '%s/MNINonLinear/fsaverage_LR32k/%s.L.midthickness_MSMAll.32k_fs_LR.surf.gii',
                            'surface_right': '%s/MNINonLinear/fsaverage_LR32k/%s.R.midthickness_MSMAll.32k_fs_LR.surf.gii',
                            'roi_left': '%s/MNINonLinear/fsaverage_LR32k/%s.L.atlasroi.32k_fs_LR.shape.gii',
                            'roi_right': '%s/MNINonLinear/fsaverage_LR32k/%s.R.atlasroi.32k_fs_LR.shape.gii'}
datasourceanat.inputs.template_args = {'seg': [['subject_id']],
                                   'surface_left': [['subject_id', 'subject_id']],
                                   'surface_right': [['subject_id', 'subject_id']],
                                   'roi_left': [['subject_id', 'subject_id']],
                                   'roi_right': [['subject_id', 'subject_id']]}
datasourceanat.inputs.sort_filelist = True


datasink = pe.Node(
    interface=nio.DataSink(),
    name="datasink")

datasink.inputs.regexp_substitutions = [
    # e.g. results/dual_regression/_subject_id_100307/_session_1/timeseries_1.csv -> 
    #   results/_subject_id_100307/_session_1/dual_regression/timeseries_1.csv
    (r'results/([a-z_]+?)/([\w/]+)/', r'results/\2/\1/'),
    (r'_subject_id_(\d+)/_session_(\d+)/', r'\1/ses-\2/'),
    (r'_subject_id_(\d+)/', r'\1/'),   
    (r'_0\.','_LR.'),
    (r'_1\.','_RL.'),
    (r'_(\w*)beta2cifti0', r'ses-1/\1beta'),
    (r'_(\w*)beta2cifti1', r'ses-2/\1beta'),
]


# ###################### #
# Preprocessing Workflow #
# ###################### #


# build workflow

getConfoundsWf = pe.Workflow(name='getconfoundswf')

inputnode = pe.Node(
    interface=util.IdentityInterface(fields=[
        'func_vol','seg']),
    name='inputspec')

# ensure data is a float
img2float = pe.Node(
    interface=fsl.ImageMaths(
        out_data_type='float', op_string='', suffix='_dtype'),
    name='img2float')

# extract csf timeseries

binarize_csf = pe.Node(Binarize(match=[4, 43]),
                         name="binarize_csf")  # Typical labels for lateral ventricles

erode_csf = pe.Node(fsl.maths.ErodeImage(kernel_shape='box',
                                         kernel_size=3),
                    name="erode_csf")

resample_csf = pe.Node(fsl.preprocess.ApplyWarp(interp='nn'),
                         name='resample_csf')

compute_csf_ts = pe.Node(fsl.utils.ImageMeants(),
    name="compute_csf_ts")


# extract wm timeseries

binarize_wm = pe.Node(Binarize(match=[2, 41]),
                         name="binarize_wm")  # Typical labels for lateral ventricles

erode_wm = pe.Node(fsl.maths.ErodeImage(kernel_shape='box',
                                         kernel_size=3),
                    name="erode_wm")

resample_wm = pe.Node(fsl.preprocess.ApplyWarp(interp='nn'),
                         name='resample_wm')

compute_wm_ts = pe.Node(fsl.utils.ImageMeants(),
    name="compute_wm_ts")


getConfoundsWf.connect([
    (inputnode, img2float, [('func_vol','in_file')]),

    # get CSF timeseries
    (inputnode, binarize_csf, [('seg', 'in_file')]),
    (binarize_csf,  erode_csf, [('binary_file', 'in_file')]),
    
    (erode_csf, resample_csf, [('out_file', 'in_file')]),
    (img2float, resample_csf, [('out_file', 'ref_file')]),

    (resample_csf, compute_csf_ts, [('out_file', 'mask')]),
    (img2float, compute_csf_ts, [('out_file', 'in_file')]),
    
    # get WM timeseries
    (inputnode, binarize_wm, [('seg', 'in_file')]),
    (binarize_wm,  erode_wm, [('binary_file', 'in_file')]),

    (img2float, resample_wm, [('out_file', 'ref_file')]),
    (erode_wm, resample_wm, [('out_file', 'in_file')]),
    
    (resample_wm, compute_wm_ts, [('out_file', 'mask')]),
    (img2float, compute_wm_ts, [('out_file', 'in_file')]),
])
    
    
# ######### #
# Utilities #
# ######### #

def pickfirst(files):
    if isinstance(files, list):
        return files[0]
    else:
	    return files

def pickfirstdeep(files):
    while isinstance(files, list):
        return pickfirstdeep(files[0])
    else:
	    return files    

def mergelists(lists):
    if isinstance(lists[0],list):
        return sum(lists,[])
    else:
        return lists

# ############ #
# Compute tSNR #
# ############ #

def init_tsnr(name='tsnr'):
    # this function produces an interface that can take one or more cifti input files,
    # computes tSNR for each, averages them acros inputs, estimates the mean tSNR
    # for each parcel specified by an input atlas, converts these to a csv file
    # of parcel tSNRs and returns that. It's designed to work equally with a 
    # scenario where tasks are spread across multiple scans or when tasks are in
    # a single scan.
    wf = pe.Workflow(name=name)

    inputnode = pe.Node(
        interface=util.IdentityInterface(
            fields=['in_files','atlas']),
        name='inputspec')

    joinWithinBlock = pe.JoinNode(
        interface=util.IdentityInterface(
            fields=['in_file']),
        joinsource='directionsource',
        joinfield=['in_file'],
        name='joinwithinblock')

    joinWithinSubject = pe.JoinNode(
        interface=util.IdentityInterface(
            fields=['in_file']),
        joinsource='sessionsource',
        joinfield=['in_file'],
        name='joinwithinsubject')

    ciftiTSNR = pe.MapNode(
        interface=wb_cifti.Reduce(
            operation="TSNR"),
        iterfield=['in_file'],
        name='ciftitsnr')

    ciftiAverage = pe.Node(
        interface=wb_cifti.Average(),
        name='ciftiaverage')

    ciftiParcellate = pe.Node(
        interface=wb_cifti.Parcellate(),
        name='ciftiparcellate')

    ciftiToText = pe.Node(
        interface=wb_cifti.CiftiConvertText(),
        name='ciftiToText')

    outputspec = pe.Node(
        interface=util.IdentityInterface(
            fields=['out_file']),
        name='outputspec')
    
    wf.connect([
        (inputnode, joinWithinBlock, [('in_files', 'in_file')]),
        (joinWithinBlock, joinWithinSubject, [('in_file', 'in_file')]),
        (joinWithinSubject, ciftiTSNR, [(('in_file', mergelists), 'in_file')]),
        
        (ciftiTSNR, ciftiAverage, [('out_file', 'in_vars')]),
        
        (inputnode, ciftiParcellate, [('atlas', 'parcellation')]),
        (ciftiAverage, ciftiParcellate, [('out_file', 'in_file')]),

        (ciftiParcellate, ciftiToText, [('out_file', 'in_file')]),

        (ciftiToText, outputspec, [('out_file', 'out_file')])
    ])

    return wf

tsnrwf = init_tsnr()

# ######################### #
# Dual Regression Worfklows #
# ###########################

dualRegWf = aligntools_wf.init_hcp_dual_regression_wf(iterations=6)

# these are used for estimating subject-to-group alignment quality
#dualRegWf.inputs.inputspec.group_ICAs_low_d = [os.path.join(data_dir,'HCP_Resources/GroupAvg/HCP_PTN1200/groupICA/groupICA_3T_HCP1200_MSMAll_d15.ica/melodic_IC.dscalar.nii'),
#                                               os.path.join(data_dir,'HCP_Resources/GroupAvg/HCP_PTN1200/groupICA/groupICA_3T_HCP1200_MSMAll_d25.ica/melodic_IC.dscalar.nii')]

# ########################### #
# run level modeling workflow #
# ########################### #

# task specific event configuration


def subjectinfo(ica_timeseries):
    import pandas as pd
    
    from nipype.interfaces.base import Bunch
    
    nodets = pd.read_csv(ica_timeseries, sep=',', header=None)
    
    ICA_names = []
    ICAs = []
    for col in nodets:
        ICA_names.append(f'ICA{col}')
        ICAs.append(nodets[col].tolist())

    ICA_names.append('constant')

    # task general variables
    output = Bunch(conditions=[],
                   onsets=[],
                   durations=[],
                   amplitudes=None,
                   tmod=None,
                   pmod=None,
                   regressor_names=ICA_names,
                   regressors=ICAs,
                   scale_regressors=False)
    
    return output

subjectinfo_node = pe.MapNode(util.Function(input_names=['ica_timeseries'],
                                 output_names=['subject_info'],
                                 function=subjectinfo),
                        iterfield=['ica_timeseries'],
                        name='subjectinfo_node')

# task specific contrast configuration

def select_contrasts(subject_info):
    # set an (unwhitened) t-stat contrast for each independent component
    contrasts = [(name, 'T', [name], [1]) for name in subject_info.regressor_names if name != 'constant']

    return contrasts

#contrastselect_node = pe.MapNode(util.Function(input_names=['subject_info'],
contrastselect_node = pe.Node(util.Function(input_names=['subject_info'],
                                            output_names=['contrasts'],
                                            function=select_contrasts),
                        #iterfield=['subject_info'],
                        name='contrastselect_node')
                        
                        

def assemble_confounds_mat(motion, csf, wm):
    import pandas as pd
    import os
    import scipy as sp

    motion_df = pd.read_csv(motion, sep='\s+', header=None)
    csf_df = pd.read_csv(csf, sep='\s+', header=None)
    wm_df = pd.read_csv(wm, sep='\s+', header=None)

    # regress 24 motion parameters and include csf and wmwhile we're at it.
    confounds = pd.concat([motion_df, motion_df**2, csf_df, wm_df], axis=1)
    
    # csf and white matter will have non-zero offsets which need to be normalized
    # to avoid reintroducing a scan effect
    confounds = sp.stats.zscore(confounds)

    cwd = os.getcwd()
    filename = os.path.join(cwd, "confounds.csv")

    confounds.to_csv(filename, sep='\t', index=False, header=False)

    return filename

assembleconfounds_node = pe.Node(util.Function(input_names=['motion','csf', 'wm'],
                                                  output_names=['confounds'],
                                                  function=assemble_confounds_mat),
                                       name='assembleconfounds_node')
 

# design and contrast configurations

inputnode_modelfitwf = pe.Node(
    interface=util.IdentityInterface(fields=[
        'func','ica_timeseries',
        'csf', 'wm', 'motion']),
    name='inputspec')


joinWithinSession = pe.JoinNode(util.IdentityInterface(
        fields=['cifti',
            'func',
            'confounds',
            'nvols']),
    joinsource='directionsource',
    joinfield=['cifti',
            'func',
            'confounds',
            'nvols'],
    name='joinwithinsession')

joinWithinSubject = pe.JoinNode(util.IdentityInterface(
        fields=['subject_info','session_info',
            'contrasts','cifti','nvols']),
    joinsource='sessionsource',
    joinfield=['subject_info','session_info',
        'contrasts','cifti','nvols'],
    name='joinwithinsubject')

# specifying parameter_source here is a hack. All other parameter sources
# get manipulated by normalize_mc_params, but if we specify the source as
# SPM then parameters simply get passed through. See nipype/utils/misc.py
designspec = pe.Node(interface=model.SpecifySPMModel(
        parameter_source='SPM',
        concatenate_runs=True), 
    name="designspec")

level1design = pe.Node(interface=spm.Level1Design(
        model_serial_correlations='FAST',
        global_intensity_normalization='none'), 
    name="level1design")
    
nvols = pe.Node(interface=NVols(),
    name="nvols")
    
def hstack(x):
    return sum(x,[])
    
fmriconcat = pe.Node(
    interface=FMRIConcat(),
    name="fmriconcat")

modelestimate = pe.Node(
    interface=spm.EstimateModel(estimation_method={'Classical': 1}),
    name='modelestimate')
    
contrastestimate = pe.Node(
    interface=spm.EstimateContrast(),
    config = {'execution': {'remove_unnecessary_outputs': False}},
    name = "contrastestimate"
)


# gzip_file provided by ChatGPT
def gzip_file(input_file, output_file=None):
    import gzip
    import shutil
    """
    Compresses a file using gzip.
    
    Parameters:
    - input_file: str, path to the file to be compressed.
    - output_file: str, path for the compressed file (optional). 
                   If not provided, it will save as input_file + '.gz'.
    
    Returns:
    - str: Path to the gzipped file.
    """
    if output_file is None:
        output_file = input_file + '.gz'

    with open(input_file, 'rb') as f_in:
        with gzip.open(output_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    
    return output_file

gzip_beta = pe.MapNode(util.Function(input_names=['input_file'],
                                                  output_names=['out_file'],
                                                  function=gzip_file),
                                       iterfield=['input_file'],
                                       name='gzipbeta')

gzip_con = pe.MapNode(util.Function(input_names=['input_file'],
                                                  output_names=['out_file'],
                                                  function=gzip_file),
                                       iterfield=['input_file'],
                                       name='gzipcon')
                                       
gzip_t = pe.MapNode(util.Function(input_names=['input_file'],
                                                  output_names=['out_file'],
                                                  function=gzip_file),
                                       iterfield=['input_file'],
                                       name='gzipt')
cifti2nifti = pe.Node(
    interface=wb_cifti.CiftiConvertNifti(
        smaller_dims=True),
    iterfield=['cifti_in'],
    name='cifti2nifti')


def select_betas_of_interest(beta_images, func, subject_info):
    # this assumes equal number of contrasts in each session
    n_run = len(func)
    n_sess = len(subject_info)
    
    # spm stacks session intercepts at the end, so we need to 
    # subract them from the count before dividing
    beta_per_sess = (len(beta_images)-n_run)/n_sess
    
    filt_beta_images = [];
    for i,info in enumerate(subject_info):
        ind0 = int(beta_per_sess*i)
        beta_names = set(info.regressor_names)
        beta_names.remove('constant') # this is the run specific intercept
        indf = int(ind0+len(beta_names))
        
        filt_beta_images.append(beta_images[ind0:indf])
        
    return filt_beta_images

selectBetasOfInterest = pe.Node(util.Function(input_names=['beta_images', 'func', 'subject_info'],
                                                 output_names=['beta_images'],
                                                 function=select_betas_of_interest),
                                       name='selectbetasofinterest')
mergebeta = pe.MapNode(
    interface=fsl.Merge(
        dimension='t'),
    iterfield=['in_files'],
    name="mergebeta")
    
beta2cifti = pe.MapNode(
    interface=wb_cifti.NiftiConvertCifti(reset_scalars=True),
    iterfield=['nifti_in'],
    name='beta2cifti')

mergetL1 = pe.Node(
    interface=fsl.Merge(
        dimension='t'),
    name="mergetl1")

tL1_2cifti = pe.Node(
    interface=wb_cifti.NiftiConvertCifti(reset_scalars=True),
    name='tL1_2cifti')
    

mergeCon = pe.Node(
    interface=fsl.Merge(
        dimension='t'),
    name="mergecon")

con2cifti = pe.Node(
    interface=wb_cifti.NiftiConvertCifti(reset_scalars=True),
    name='con2cifti')
    
modelFitWf = pe.Workflow(name='modelfitwf')

modelFitWf.connect([
    # set up first level GLMs
    (inputnode_modelfitwf, subjectinfo_node, [('ica_timeseries', 'ica_timeseries')]),
    (subjectinfo_node, contrastselect_node, [(('subject_info', pickfirst), 'subject_info')]),

    (inputnode_modelfitwf, assembleconfounds_node, [('csf','csf'),
                                                  ('wm','wm'),
                                                  ('motion','motion')]),

    (inputnode_modelfitwf, cifti2nifti, [('func','cifti_in')]),       
    
    (cifti2nifti, joinWithinSession, [('out_file', 'func')]),
    (assembleconfounds_node, joinWithinSession, [('confounds', 'confounds')]),

    # we can't have spike regressors because they won't be matched across runs and
    # we won't be able to compute crossvalidated RSA metrics
    (joinWithinSession, designspec, [('func', 'functional_runs')]),
    (subjectinfo_node, designspec, [('subject_info','subject_info')]),
    (joinWithinSession, designspec, [('confounds', 'realignment_parameters')]),
    
    (designspec, joinWithinSubject, [(('session_info', pickfirst), 'session_info')]),
    (contrastselect_node, joinWithinSubject, [('contrasts', 'contrasts')]),

    (joinWithinSubject, level1design, [('session_info', 'session_info')]),
    (joinWithinSubject, contrastestimate, [(('contrasts', pickfirst), 'contrasts')]),
    
    # separate timeseries models between concatenated scans
    (cifti2nifti, nvols, [('out_file', 'in_file')]),
    (nvols, joinWithinSession, [('nvols', 'nvols')]),
    (joinWithinSession, joinWithinSubject, [('nvols', 'nvols')]),
    (joinWithinSubject, fmriconcat, [(('nvols', hstack), 'nscan')]),
    (level1design, fmriconcat, [('spm_mat_file', 'spm_mat_file')]),
    
    # run first level GLMs
    # add a call to spm_fmri_concatenate here to modify the spm_mat_file
    (fmriconcat, modelestimate, [('spm_mat_file', 'spm_mat_file')]),
    (modelestimate, contrastestimate, [('spm_mat_file', 'spm_mat_file'),
                                       ('beta_images', 'beta_images'),
                                       ('residual_image', 'residual_image')]),
                                       
    (modelestimate, gzip_beta, [('beta_images', 'input_file')]),
    (contrastestimate, gzip_con, [('con_images', 'input_file')]),
    (contrastestimate, gzip_t, [('spmT_images', 'input_file')]),
    
    (inputnode_modelfitwf, joinWithinSession, [('func','cifti')]),  
    (joinWithinSession, joinWithinSubject, [('cifti', 'cifti')]),

    # merge betas
    # first merge [sess1_run1, sess1_run2, sess2_run1, sess2_run2] -> 
    #  -> [([sess1_run1, sess1_run2], pickfirst), ([sess2_run1, sess2_run2], pickfirst)] -> 
    #  -> [sess1_run1, sess2_run1]
    # We use these to extract regressor columns within session blocks
    (subjectinfo_node, joinWithinSubject, [(('subject_info', pickfirst), 'subject_info')]),
    (joinWithinSubject, selectBetasOfInterest, [('subject_info', 'subject_info')]),
    (joinWithinSubject, selectBetasOfInterest, [(('cifti', hstack), 'func')]), # we use this to determine the number of intercepts
    (modelestimate, selectBetasOfInterest, [('beta_images', 'beta_images')]),
    
    (selectBetasOfInterest, mergebeta, [('beta_images', 'in_files')]),
    (mergebeta, beta2cifti, [('merged_file', 'nifti_in')]),
    (joinWithinSubject, beta2cifti, [(('cifti', pickfirstdeep), 'cifti_template')]),

    # merge tstats
    (contrastestimate, mergetL1, [('spmT_images', 'in_files')]),
    (mergetL1, tL1_2cifti, [('merged_file', 'nifti_in')]),
    (joinWithinSubject, tL1_2cifti, [(('cifti', pickfirstdeep), 'cifti_template')]),
    
    # merge contrasts
    (contrastestimate, mergeCon, [('con_images', 'in_files')]),
    (mergetL1, con2cifti, [('merged_file', 'nifti_in')]),
    (joinWithinSubject, con2cifti, [(('cifti', pickfirstdeep), 'cifti_template')])
])

##########
# do RSA #
##########

def init_betas(name='whitenbetas', normmethod='multivariate', normmode='runwise'):
    
    betaswf = pe.Workflow(name=name)
    
    inputnode_betas = pe.Node(
        interface=util.IdentityInterface(fields=[
            'spm_mat_file','nifti_atlas', 'cifti_template', 'func', 'subject_info']),
        name='inputspec')

    betas = pe.Node(
        interface=SpatialWhitening(
            normmode=normmode),
        name="betas")
    if normmethod == 'univariate':
        betas.inputs.shrinkage = 1.0
        
    splitbetas = pe.Node(
        interface=fsl.Split(dimension='t'),
        name="splitbetas")

    selectBetasOfInterest = pe.Node(util.Function(input_names=['beta_images', 'func', 'subject_info'],
                                                    output_names=['beta_images'],
                                                    function=select_betas_of_interest),
                                        name='selectbetasofinterest')
    mergebetas = pe.MapNode(
        interface=fsl.Merge(
            dimension='t'),
        iterfield=['in_files'],
        name="mergebetas")
        
    beta2cifti = pe.MapNode(
        interface=wb_cifti.NiftiConvertCifti(reset_scalars=True),
        iterfield=['nifti_in'],
        name='beta2cifti')

    outputnode_betas = pe.Node(
        interface=util.IdentityInterface(fields=['betas']),
        name='outputspec')

    betaswf.connect([        
        (inputnode_betas, betas, [('spm_mat_file', 'spm_mat_file'),
                                  ('nifti_atlas', 'atlas')]),
            
        # split whitened betas by session, merge and convert to cifti
        (betas, splitbetas, [('whitened_images', 'in_file')]),
        (splitbetas, selectBetasOfInterest, [
            ('out_files', 'beta_images')]),
        (inputnode_betas, selectBetasOfInterest, [
            ('subject_info', 'subject_info'),
            ('func', 'func')]),
        
        (selectBetasOfInterest, mergebetas, [('beta_images', 'in_files')]),
        (mergebetas, beta2cifti, [('merged_file', 'nifti_in')]),
        (inputnode_betas, beta2cifti, [('cifti_template', 'cifti_template')]),

        (beta2cifti, outputnode_betas, [('out_file', 'betas')])
    ])

    return betaswf


def init_rsawf(name='rsa',normmethod='multivariate', save_whitening_matrix=False):

    rsawf = pe.Workflow(name=name)

    # func should be a list containing all input functional runs that went into creating
    # subject_info. It's used to discount intercepts when finding betas of interest.
    inputnode_rsa = pe.Node(
        interface=util.IdentityInterface(fields=[
            'spm_mat_file','atlas', 'func', 'subject_info']),
        name='inputspec')

    atlas2nifti = pe.Node(
        interface=wb_cifti.CiftiConvertNifti(
            smaller_dims=True),
        iterfield=['cifti_in'],
        name='atlas2nifti')

    rsa = pe.Node(
        interface=RDM(
            normmethod=normmethod,
            save_whitening_matrix=save_whitening_matrix),
        name="rsa")

    betas_l1 = init_betas(name='betas_l1', normmethod=normmethod, normmode='runwise')
    
    betas_l2 = init_betas(name='betas_l2', normmethod=normmethod, normmode='overall')

    def getCiftiMathsExpression(stats):
            return '('.join([f'var{i} + ' for i in range(len(stats))]) + f' 0)/{len(stats)}'
        
    def getInVars(stats):
            return [(f'var{i}', s) for i,s in enumerate(stats)]
        
    averageBetas = pe.Node(
        interface=wb_cifti.CiftiMath(),
        name="averagebetas")

    cosim = pe.Node(
        interface=WithinSimilarity(
            normmethod=normmethod),
        name='cosim')

    outputnode_rsa = pe.Node(
        interface=util.IdentityInterface(fields=[
            'rdm','whitened_rdm','whitening_matrix','whitening_matrix_metadata',
            'cosim','betanames', 'betas_l1','betas_l2']),
        name='outputspec')

    rsawf.connect([
        (inputnode_rsa, atlas2nifti, [('atlas', 'cifti_in')]),

        (inputnode_rsa, rsa, [('spm_mat_file', 'spm_mat_file')]),
        (atlas2nifti, rsa, [('out_file', 'atlas')]),
        
        (inputnode_rsa, betas_l1, [
            ('spm_mat_file','inputspec.spm_mat_file'),
            ('atlas', 'inputspec.cifti_template'),
            ('func','inputspec.func'),
            ('subject_info','inputspec.subject_info')]),
        (atlas2nifti, betas_l1, [('out_file', 'inputspec.nifti_atlas')]),
        (inputnode_rsa, betas_l2, [
            ('spm_mat_file','inputspec.spm_mat_file'),
            ('atlas', 'inputspec.cifti_template'),
            ('func','inputspec.func'),
            ('subject_info','inputspec.subject_info')]),
        (atlas2nifti, betas_l2, [('out_file', 'inputspec.nifti_atlas')]),
        (inputnode_rsa, cosim, [('spm_mat_file', 'spm_mat_file')]),
        (atlas2nifti, cosim, [('out_file', 'atlas')]),

        (rsa, outputnode_rsa, [('rdm', 'rdm'),
                               ('whitened_rdm', 'whitened_rdm'),
                               ('whitening_matrix', 'whitening_matrix'),
                               ('whitening_matrix_metadata', 'whitening_matrix_metadata')]),
        (betas_l1, outputnode_rsa, [('outputspec.betas', 'betas_l1')]),
        (betas_l2, averageBetas, [(('outputspec.betas', getInVars), 'in_vars'),
                                  (('outputspec.betas', getCiftiMathsExpression), 'expression')]),
        (averageBetas, outputnode_rsa, [('out_file', 'betas_l2')]),
        (cosim, outputnode_rsa, [('similarity','cosim'),
                                 ('betanames','betanames')]),
    ])

    return rsawf

stddistwf = init_rsawf(name='stddist',normmethod='univariate', save_whitening_matrix=True)
crossnobiswf = init_rsawf(name='crossnobis', save_whitening_matrix=True)


# ############################################## #
# connect preproc, run levels and subject levels #
# ############################################## #

# this joins LR and RL runs/metadata within session
joinFuncWithinSession = pe.JoinNode(util.IdentityInterface(
                        fields=['func']),
                        joinsource='directionsource',
                        joinfield=['func'],
                        name='joinfuncwithinsession')

subjectlevel = pe.Workflow(name='subjectlevel')
subjectlevel.connect([
    (subjectsource, datasourceanat, [('subject_id', 'subject_id')]),
    (subjectsource, datasourcefunc, [('subject_id', 'subject_id')]),
    (sessionsource, datasourcefunc, [('session','session')]),
    (directionsource, datasourcefunc, [('direction','direction')]),
    
    (datasourceanat, getConfoundsWf, [
        ('seg', 'inputspec.seg')]),
    (datasourcefunc, getConfoundsWf, [
        ('vol', 'inputspec.func_vol')]),

    (datasourcefunc, tsnrwf, [('func', 'inputspec.in_files')]),

    (datasourceanat, dualRegWf, [
        ('surface_left', 'inputspec.surface_left'),
        ('surface_right', 'inputspec.surface_right'),
        ('roi_left', 'inputspec.roi_left'),
        ('roi_right', 'inputspec.roi_right')]),
    (datasourcefunc, joinFuncWithinSession, [('func', 'func')]),
    (joinFuncWithinSession, dualRegWf, [('func', 'inputspec.func')]),

    (datasourcefunc, modelFitWf, [
        ('motion', 'inputspec.motion'),
        ('func', 'inputspec.func')]),
    (getConfoundsWf, modelFitWf, [        
        ('compute_csf_ts.out_file', 'inputspec.csf'),
        ('compute_wm_ts.out_file', 'inputspec.wm')]),
    (dualRegWf, modelFitWf, [('outputspec.timeseries', 'inputspec.ica_timeseries')]),

    (modelFitWf, stddistwf, [('modelestimate.spm_mat_file', 'inputspec.spm_mat_file'),
                         ('joinwithinsubject.subject_info', 'inputspec.subject_info'),
                         (('joinwithinsubject.cifti', hstack), 'inputspec.func')]),

    (modelFitWf, crossnobiswf, [('modelestimate.spm_mat_file', 'inputspec.spm_mat_file'),
                         ('joinwithinsubject.subject_info', 'inputspec.subject_info'),
                         (('joinwithinsubject.cifti', hstack), 'inputspec.func')]),

    (tsnrwf, datasink, [('outputspec.out_file', 'results.@tsnr')]),

    (dualRegWf, datasink, [('outputspec.timeseries', 'results.dual_regression.@ica_timeseries'),
                          ('outputspec.betas', 'results.dual_regression.@beta'),
                          ('outputspec.tstats', 'results.dual_regression.@tstats'),
                          ('outputspec.zstats', 'results.dual_regression.@zstats'),
                          ('outputspec.mean_tstat', 'results.dual_regression.@mean_tstat')]),

    (modelFitWf, datasink, [('gzipbeta.out_file', 'results.spm.@beta_images'),
                            ('modelestimate.residual_image', 'results.spm.@residual_image'),
                            ('modelestimate.RPVimage', 'results.spm.@RPVimage'),
                            ('modelestimate.mask_image', 'results.spm.@mask_image'),
                            ('gzipcon.out_file', 'results.spm.@con_images'),
                            ('gzipt.out_file', 'results.spm.@spmT_images'),
                            ('contrastestimate.spm_mat_file', 'results.spm.@spm_mat_file'),
                            ('tL1_2cifti.out_file', 'results.@tstats'),
                            ('beta2cifti.out_file', 'results.@merged_cifti_betas'),
                            ('con2cifti.out_file', 'results.@con')
                            ]),

    (stddistwf, datasink, [('outputspec.rdm', 'results.standardized_betas.@stddist'),
                       ('outputspec.whitened_rdm', 'results.standardized_betas.@whitened_stddist'),
                       ('outputspec.whitening_matrix', 'results.standardized_betas.@whitening_matrix'),
                       ('outputspec.whitening_matrix_metadata', 'results.standardized_betas.@whitening_matrix_metadata'),
                       ('outputspec.betas_l1', 'results.@merged_cifti_stdbetas'),
                       ('outputspec.cosim', 'results.standardized_betas.@cosim'),
                       ('outputspec.betas_l2', 'results.standardized_betas.@mean_betas'),
                       ('outputspec.betanames', 'results.standardized_betas.@betanames')]),


    (crossnobiswf, datasink, [('outputspec.rdm', 'results.whitened_betas.@stddist'),
                       ('outputspec.whitened_rdm', 'results.whitened_betas.@whitened_stddist'),
                       ('outputspec.whitening_matrix', 'results.whitened_betas.@whitening_matrix'),
                       ('outputspec.whitening_matrix_metadata', 'results.whitened_betas.@whitening_matrix_metadata'),
                       ('outputspec.betas_l1', 'results.@merged_cifti_whitenedbetas'),
                       ('outputspec.cosim', 'results.whitened_betas.@cosim'),
                       ('outputspec.betas_l2', 'results.whitened_betas.@mean_betas'),
                       ('outputspec.betanames', 'results.whitened_betas.@betanames')]),

])


subjectlevel.inputs.modelfitwf.designspec.input_units = 'secs'
subjectlevel.inputs.modelfitwf.designspec.time_repetition = TR
subjectlevel.inputs.modelfitwf.designspec.high_pass_filter_cutoff = hp_cutoff

subjectlevel.inputs.modelfitwf.level1design.interscan_interval = TR
subjectlevel.inputs.modelfitwf.level1design.timing_units = 'secs'
subjectlevel.inputs.modelfitwf.level1design.bases = {'hrf': {'derivs': [0,0]}}

# ############# #
# main function #
# ############# #

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="HCP dual regression estimation")
    parser.add_argument('--subject_ids', nargs='*', help="Subject ID. Must match HCP directory name")
    parser.add_argument('--out', type=str, required=True, help='Output basename. \
                        Outputs will be in subfolders labed by subject_id, task and direction.')
    parser.add_argument('--scratch', type=str, required=True,
                        help='scratch directory where temporary files should be stored')
    parser.add_argument('--n_cpus', type=int, default=int(os.getenv('SLURM_CPUS_PER_TASK',default='1')),
                        help='Number of CPUs available for computation')
    parser.add_argument('--atlas', required=True,
                        help='Parcellation to use for RSA. RDMs are computed for each parcel and spatial whitening is done within parcels')
    parser.add_argument('--rsn_template', required=True,
                        help='HCP Group ICA to dual regress onto subject data. Alignment weighing will be invalid for non-HCP group ICAs.')
    parser.add_argument('--data_dir', type=str, required=False,
                        help='Path to HCP data directory immediately above subject folders, e.g. HCP1200')
    parser.add_argument('--hcp_resources_dir', type=str, required=False, default=os.path.abspath('/dartfs/rc/lab/D/DBIC/DBIC/archive/HCP/'),
                        help='Path to HCP Resources directory. This should contain the GroupAvg/HCP_PTN1200 subfolders.')

    args = parser.parse_args()

    if args.data_dir is not None:
        datasourcefunc.inputs.base_directory = args.data_dir
        datasourceanat.inputs.base_directory = args.data_dir

    subjectsource.iterables = [('subject_id', args.subject_ids)]
    sessionsource.iterables = [('session', [1,2])]
    directionsource.iterables = [('direction', ['LR','RL'])]

    SCRATCH_DIR = args.scratch
    subjectlevel.base_dir = os.path.abspath(SCRATCH_DIR + '/workingdir')
    subjectlevel.config = {
        "execution": {
            "crashdump_dir": os.path.abspath(SCRATCH_DIR + '/crashdumps')
        }
    }

    datasink.inputs.base_directory = os.path.abspath(args.out)
    
    subjectlevel.inputs.stddist.inputspec.atlas = os.path.abspath(args.atlas)
    subjectlevel.inputs.crossnobis.inputspec.atlas = os.path.abspath(args.atlas)
    subjectlevel.inputs.tsnr.inputspec.atlas = args.atlas
    
    subjectlevel.inputs.dualregressionwf.inputspec.group_ICA = os.path.join(args.rsn_template)
    subjectlevel.inputs.dualregressionwf.inputspec.group_ICAs_low_d = [os.path.join(args.hcp_resources_dir,'HCP_Resources/GroupAvg/HCP_PTN1200/groupICA/groupICA_3T_HCP1200_MSMAll_d15.ica/melodic_IC.dscalar.nii'),
                                               os.path.join(args.hcp_resources_dir,'HCP_Resources/GroupAvg/HCP_PTN1200/groupICA/groupICA_3T_HCP1200_MSMAll_d25.ica/melodic_IC.dscalar.nii')]


    subjectlevel.write_graph()
    if args.n_cpus and args.n_cpus > 1:
        print(f'Multithreading with {args.n_cpus} CPUs')
        outgraph = subjectlevel.run(plugin='MultiProc', plugin_args={'n_procs':args.n_cpus})
    else:
        outgraph = subjectlevel.run()
