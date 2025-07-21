from nipype.interfaces.matlab import MatlabCommand
from nipype.interfaces.base import (
    TraitedSpec,
    BaseInterface,
    BaseInterfaceInputSpec,
    File,
)
from traits.api import List, Bool, Float, Enum, Either
import os
from string import Template


class SpatialWhiteningInputSpec(BaseInterfaceInputSpec):
    spm_mat_file = File(exists=True, mandatory=True, 
        desc="path to SPM file in main SPM directory (containing the betas)")

    atlas = File(exists=True, mandatory=False,
        desc="Path to an atlas in register with SPM betas")
        
    normmode = Enum('overall','partwise','runwise',
        usedefault=True,
        desc="Do multivariate noise normalization by run or overall") 
        
    shrinkage = Float(-1,
        usedefault=True,
        desc="1 means diagonal spatial covariance. Default is to use Ledoit-Wolf method to pick optimal factor."
    )
        

class SpatialWhiteningOutputSpec(TraitedSpec):
    whitened_images = File(exists=True)


class SpatialWhitening(BaseInterface):
    """
    Uses the rsatoolbox to implement spatial whitening based on the optimally
    regularized spatial covariance matrix. Regularization is estimated according
    to the method of Ledoit and Wolf (2004) Journal of Portfolio Management.
    You will need spm12 and the rsatoolbox on your path, which can be obtained here:
    https://github.com/rsagroup/rsatoolbox_matlab. The images differ from 
    multivariate t-stats because they don't take the design covariance structure
    into account. For t-stat computations refer to:
    https://www.fil.ion.ucl.ac.uk/spm/doc/books/hbf2/pdfs/Ch8.pdf

    Note that the atlas should be an indexed map in the same space as the one 
    in which you're running spm. So if you're running SPM on surface data that's
    been converted to nifti cubes, make sure you convert your atlas the same way too.
    Covariance estimation is spatially agnostic (i.e. there's no distance based
    taper in spatial correlation or anything like that), it's all purely data driven 
    so you don't need to be working in any meaningful anatomical space.
    """

    input_spec = SpatialWhiteningInputSpec
    output_spec = SpatialWhiteningOutputSpec

    def _run_interface(self, runtime):    
        d = dict(spm_mat_file=self.inputs.spm_mat_file,
                 atlas=self.inputs.atlas,
                 normmode=self.inputs.normmode,
                 shrinkage=self.inputs.shrinkage,
                 whitened_images='beta_whitened.nii')

        # This is your MATLAB code template
        script = Template(
            """ spm_mat_file = '$spm_mat_file';
                atlas_path = '$atlas';
                normmode = '$normmode';
                whitened_images = '$whitened_images';
                shrinkage = $shrinkage;
                
                if shrinkage > 0
                    varg = {'shrinkage', shrinkage, 'normmode', normmode};
                else
                    varg = {'normmode', normmode};
                end

                % import atlas
                atlas_hdr = spm_vol(atlas_path);
                atlas_vols = spm_read_vols(atlas_hdr);
                [x0, y0, z0, t0] = size(atlas_vols);
                atlas = reshape(atlas_vols, x0*y0*z0, t0);

                if t0 > 1
                    error('Multivariate noise normalization is only supported for 3d atlases. Consider fslsplitting atlas and running this as a mapnode instead');
                end

                % import data
                SPM = importdata(spm_mat_file);

                filename = {};
                for i = 1:size(SPM.xY.P,1)
                    str = strsplit(SPM.xY.P(i,:),','); 
                    filename{end+1} = str{1};
                end
                filename = unique(filename(:));
                filename = cat(1,filename{:});

                vols = cell(1,length(filename));
                for i = 1:size(filename,1)
                    vols{i} = niftiread(filename(i,:)); 
                end
                vols = cat(4,vols{:});

                %{
                for i = 1:size(SPM.xY.P,1)
                    hdr(i) = spm_vol(SPM.xY.P(i,:)); 
                end
                vols = spm_read_vols(hdr);
                %}
                [x,y,z,t] = size(vols);

                if x ~= x0 || y ~= y0 | z ~= z0
                    error('Atlas and data dimensions are mismatched.');
                end

                Y = double(reshape(vols, x*y*z, t))';

                % loop over unique atlas regions and whiten each parcel independently
                newMap0 = zeros(length(SPM.Vbeta), size(atlas,1));
                uniq_rois = unique(atlas(:));
                uniq_rois(uniq_rois == 0) = [];
                for i = 1:length(uniq_rois)
                    this_roi = uniq_rois(i);
                    roi = any(this_roi == atlas, 2); % atlas might be overlapping searchlights across multiple volumes
                    beta = rsa.spm.noiseNormalizeBeta(Y(:,roi), SPM, varg{:});

                    newMap0(:,atlas == this_roi) = beta;
                end

                newMap = reshape(newMap0', x0, y0, z0, length(SPM.Vbeta));
                newMap(isnan(newMap)) = 0;

                nii_path = strrep(whitened_images,'.nii.gz','.nii');
                niftiwrite(newMap, nii_path);
                gzip(nii_path);
                delete(nii_path);
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

        self.whitened_images = os.path.abspath(d['whitened_images'] + '.gz')

        return result.runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['whitened_images'] = os.path.abspath(self.whitened_images)
        return outputs



# this code computes a wihtin subject contrast cosine similarity metric
# by splitting data into training/test runs, computing contrasts for each
# and estimating cosine similarity between the two. Results are averaged
# across all available splits
class WithinSimilarityInputSpec(BaseInterfaceInputSpec):
    spm_mat_file = File(exists=True, mandatory=True, 
        desc="Paths to SPM.mat file produce by EstimateModel")

    atlas = File(exists=True, mandatory=False,
        desc="Path to an atlas in register with SPM betas")
        
    normmethod = Enum('multivariate', 'univariate', 'none',
        usedefault=True,
        desc="spatial normalization method to use")

    shrinkage = Float(None,
        desc="If set, shrinkage is skipped and this value is used instead")

    target = Enum('diagonal','scaledidentity',
        usedefault=True,
        desc = "What shrinkage target (prior) to use during covariance normalization.")

    nonlinearshrink = Bool(False,
        usedefault=True,
        desc = "Use nonlinear shrinkage for p > 50, n > 50.")

    # Safest default is runwise, but setting to partwise for backwards compatibility
    normmode = Enum('partwise', 'runwise', 'overall',
        usedefault=True,
        desc="Do multivariate noise normalization by run or overall") 

class WithinSimilarityOutputSpec(TraitedSpec):
    similarity = File(exists=True)

    betanames = File(exists=True)

class WithinSimilarity(BaseInterface):
    input_spec = WithinSimilarityInputSpec
    output_spec = WithinSimilarityOutputSpec

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
                 normmode=self.inputs.normmode,
                 shrinkage=self.inputs.shrinkage,
                 target=self.inputs.target,
                 nonlinearshrink=int(bool(self.inputs.nonlinearshrink)),
                 spm_mat_file=self.inputs.spm_mat_file,
                 out=out,
                 names_out='betanames.csv')

        script = Template(
            """ spm_mat_file = '$spm_mat_file';
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

                if x ~= x0 || y ~= y0 | z ~= z0
                    error('Atlas and data dimensions are mismatched.');
                end

                Y = double(reshape(vols, x*y*z, t))';


                % set up condition vectors
                X = SPM.xX.X;

                conditions = zeros(size(X,2),1);
                gSF = SPM.xGX.gSF;
                for i = 1:length(SPM.Sess)
                    sess_ind = SPM.Sess(i).col;
                    con_ind = find(contains(SPM.xX.name(sess_ind),{'Task-','ICA'}) & ~contains(SPM.xX.name(sess_ind),'Cue'));
                    conditions(sess_ind(con_ind)) = 1:length(con_ind);
                end


                % loop over unique atlas regions and compute distance vectors
                uniq_rois = unique(atlas(:));
                uniq_rois(uniq_rois == 0) = [];
                similarity = nan(sum(unique(conditions)>0), sum(unique(atlas(:))>0));
                ncon = size(similarity,1);
                fun=@(x1,x2)(x1(:)'*x2(:)/(norm(x1(:))*norm(x2(:)))); % cosine similarity metric
                for i = 1:length(uniq_rois)
                    this_roi = uniq_rois(i);
                    roi = any(this_roi == atlas, 2); % atlas might be overlapping searchlights across multiple volumes

                    this_Y = Y(:,roi).*gSF; % mask and apply SPM global signal scaling;
                    this_Y = this_Y(:,var(this_Y) > 0);
                    if any(this_Y)
                        [similarity(:,i), names] = betweenSessionSimilarity(this_Y, SPM, conditions(:), fun, ...
                            'normmethod', normmethod, 'normmode', normmode, ...
                            'shrinkage', shrinkage, 'target', target,'nonlinearshrink',nonlinearshrink);
                    end
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


class SpatialWhiteningMultiTaskInputSpec(BaseInterfaceInputSpec):
    spm_mat_files = Either(List(File(exists=True)), File(exists=True),
        mandatory=True, 
        desc="path to SPM file in main SPM directory (containing the betas)")

    atlas = File(exists=True, mandatory=False,
        desc="Path to an atlas in register with SPM betas")
        
    normmode = Enum('overall','partwise','runwise','poolparts','poolruns',
        usedefault=True,
        desc="Do multivariate noise normalization by run or overall") 
        
    shrinkage = Float(-1,
        usedefault=True,
        desc="1 means diagonal spatial covariance. Default is to use Ledoit-Wolf method to pick optimal factor."
    )
        

class SpatialWhiteningMultiTaskOutputSpec(TraitedSpec):
    whitened_images = File(exists=True)

    betanames = File(exists=True)


class SpatialWhiteningMultiTask(BaseInterface):
    """
    Uses the rsatoolbox to implement spatial whitening based on the optimally
    regularized spatial covariance matrix. Regularization is estimated according
    to the method of Ledoit and Wolf (2004) Journal of Portfolio Management.
    You will need spm12 and the rsatoolbox on your path, which can be obtained here:
    https://github.com/rsagroup/rsatoolbox_matlab. The images differ from 
    multivariate t-stats because they don't take the design covariance structure
    into account. For t-stat computations refer to:
    https://www.fil.ion.ucl.ac.uk/spm/doc/books/hbf2/pdfs/Ch8.pdf

    Note that the atlas should be an indexed map in the same space as the one 
    in which you're running spm. So if you're running SPM on surface data that's
    been converted to nifti cubes, make sure you convert your atlas the same way too.
    Covariance estimation is spatially agnostic (i.e. there's no distance based
    taper in spatial correlation or anything like that), it's all purely data driven 
    so you don't need to be working in any meaningful anatomical space.
    """

    input_spec = SpatialWhiteningMultiTaskInputSpec
    output_spec = SpatialWhiteningMultiTaskOutputSpec

    def _run_interface(self, runtime):    
        import numpy as np

        d = dict(atlas=self.inputs.atlas,
                 normmode=self.inputs.normmode,
                 shrinkage=self.inputs.shrinkage,
                 whitened_images='beta_whitened.nii',
                 names_out='betanames.csv')

        # I don't know how to pass a list into the string Template, so instead
        # I save these to a txt and reimport them.
        if isinstance(self.inputs.spm_mat_files, list):
            paths = np.array(self.inputs.spm_mat_files)
        else:
            paths = np.array([self.inputs.spm_mat_files])
        np.savetxt('spm_mat_files.csv', paths, fmt="%s", delimiter="\n")

        # This is your MATLAB code template
        script = Template(
            """ spm_mat_files = textread('spm_mat_files.csv','%s\\n');
                atlas_path = '$atlas';
                normmode = '$normmode';
                whitened_images = '$whitened_images';
                shrinkage = $shrinkage;
                
                if shrinkage > 0
                    varg = {'shrinkage', shrinkage, 'normmode', normmode};
                else
                    varg = {'normmode', normmode};
                end

                % import atlas
                atlas_hdr = spm_vol(atlas_path);
                atlas_vols = spm_read_vols(atlas_hdr);
                [x0, y0, z0, t0] = size(atlas_vols);
                atlas = reshape(atlas_vols, x0*y0*z0, t0);

                if t0 > 1
                    error('Multivariate noise normalization is only supported for 3d atlases. Consider fslsplitting atlas and running this as a mapnode instead');
                end

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

                % loop over unique atlas regions and whiten each parcel independently
                numBeta = 0;
                for i = 1:length(SPM)
                    numBeta = numBeta + length(SPM{i}.Vbeta);
                end
                newMap0 = zeros(numBeta, size(atlas,1));
                uniq_rois = unique(atlas(:));
                uniq_rois(uniq_rois == 0) = [];
                for i = 1:length(uniq_rois)
                    this_roi = uniq_rois(i);
                    roi = any(this_roi == atlas, 2); % atlas might be overlapping searchlights across multiple volumes
                    [beta,names] = noiseNormalizeBetaMultiTask(Y(:,roi), SPM, varg{:});

                    newMap0(:,atlas == this_roi) = beta;
                end

                newMap = reshape(newMap0', x0, y0, z0, numBeta);
                newMap(isnan(newMap)) = 0;

                nii_path = strrep(whitened_images,'.nii.gz','.nii');
                niftiwrite(newMap, nii_path);
                gzip(nii_path);
                delete(nii_path);

                fid = fopen('$names_out','w+');
                names = names(:);
                fprintf(fid, '%s\\n', names{:});
                fclose(fid)
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

        self._whitened_images = os.path.abspath(d['whitened_images'] + '.gz')
        self._betanames = os.path.abspath(d['names_out'])

        return result.runtime

    def _list_outputs(self):
        return {
            'whitened_images': self._whitened_images,
            'betanames': self._betanames
        }
