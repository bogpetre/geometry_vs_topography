import nibabel as ni
from nibabel.cifti2 import Cifti2Header, ScalarAxis
from nibabel.cifti2.cifti2 import Cifti2Image

import numpy as np
import glob
import scipy as sp

from nipype.interfaces.base import BaseInterface, BaseInterfaceInputSpec, traits, File, TraitedSpec, isdefined
from traits.api import List

from ..utils import cifti_utils
from ..ica.dual_regression import model as dr


class estimateAlignmentMapInputSpec(BaseInterfaceInputSpec):
    in_files=List(File(),
        desc="List of 4D CIFTI or NIFTI files containing connectome or timeseries data to be aligned",
        exists=True,
        required=True)

    group_maps=List(File(),
        desc="List of 4D CIFTI or NIFTI file containing ICA data to aligned to. If multiple files are \
            provided, timeseries data is dual regressed on each, alignment is evaluated for each and \
            alignment map returned is the average of these.",
        exists=True,
        required=True)
        
    weights=File(None,
        desc="CIFTI or NIFTI file of distortion weights",
        exists=True,
        required=False)
        
    zscorets=traits.Bool(False,
        usedefault=True,
        desc="Whether to estimate standardized timseries coefficients")

    zscoremap=traits.Bool(False,
        usedefault=True,
        desc="Whether to estimate standardized spatial coefficients")
        

class estimateAlignmentMapOutputSpec(TraitedSpec):
    out_file=File(
        desc="Path of output data.",
        exists=True)


class estimateAlignmentMap(BaseInterface):
    input_spec = estimateAlignmentMapInputSpec
    output_spec = estimateAlignmentMapOutputSpec
    
    
    def _run_interface(self, runtime):

        img = [ni.load(c) for c in self.inputs.in_files]
        groupMapsImg = [ni.load(gm) for gm in self.inputs.group_maps]
        if isinstance(img[0], Cifti2Image):
            # import
            data = [c.get_fdata() for c in img]
            groupMaps = [gm.get_fdata() for gm in groupMapsImg]
        
            # mask because zero-var vertices will return nans when computing
            # alignments with pearson correlation, but mainly for consistency
            # with volumetric analyses where we need to mask a bunch of 
            # extracerebral data for computational tractability
            mask = np.any([np.any(d != 0, keepdims=True, axis=0) for d in data], axis=0)
            mask = np.squeeze(mask)
            
            if self.inputs.weights:
                # the distortion map is a measure of variation in distances between vetices
                # on a mesh surface, so it's only defined on the surface
                surface_mask = img[0].header.get_axis(1).surface_mask
                distortionMap = np.ones((1, data[0].shape[1]))
                distortionMap[:,surface_mask] = ni.load(self.inputs.weights).get_fdata()

        elif isinstance(img[0], ni.nifti1.Nifti1Image):
            # import and vectorize
            data = [img[0].get_fdata()]
            x, y, z, t = data[0].shape
            data[0] = data[0].reshape(x*y*z,t).T

            data = data + [c.get_fdata().reshape(x*y*z,t).T for c in img[1:]]
            groupMaps = [gm.get_fdata().reshape(x*y*z,-1).T for gm in groupMapsImg]

            # implicit masking. If there's been any masking during preprocessing
            # then we'll have a bunch of empty extracerebral (and possibly intracerebral)
            # space here that we can ignore for efficiency.
            mask = np.any([np.any(d != 0, keepdims=True, axis=0) for d in data], axis=0)
            mask = np.squeeze(mask)

            if self.inputs.weights:
                distortionMap = ni.load(self.inputs.weights).get_fdata().reshape(x,y,z,-1).T

        else:
            raise TypeError(f'Only Nift1Image and Cifti2Image objects are supported, but {self.inputs.in_files[0]} is type {type(img[0])}')

        data = [d[:,mask] for d in data]
        groupMaps = [gm[:,mask] for gm in groupMaps]

        if self.inputs.weights:
            distortionMap = distortionMap[:,mask]

            distortion = np.ones((1, sum(mask)))
            distortion[0,:sum(mask)] = distortionMap / np.mean(distortionMap)
        else:
            distortion=None

        # do an initial dual regression and compute pearson correlation between results
        # and ICA maps vertex-wise. Each vertex in alignmentMap then tells us how correlated
        # network weights are with the template. Vertices that show higher correlations across
        # network maps are better aligned.
        alignmentMap = np.zeros((1, mask.shape[0]))
        alignmentMap[0, mask] = dr.get_alignment_map(
            data, groupMaps, weights = distortion, 
            zscorets = self.inputs.zscorets, zscoremap = self.inputs.zscoremap)

        outputs = self._list_outputs()
        if isinstance(img[0], Cifti2Image):
            template_cifti = groupMapsImg[0]
            brain_model_axis = cifti_utils.get_brain_model_axis(template_cifti)

            lbls = [str(int(ind)) for ind in range(0, alignmentMap.shape[0])]
            scalar_axis = ScalarAxis(name=np.array(lbls))

            hdr = Cifti2Header.from_axes((scalar_axis, brain_model_axis))

            alignmentMap = ni.Cifti2Image(alignmentMap, header=hdr, nifti_header=template_cifti.nifti_header)
        elif isinstance(img[0], ni.nifti1.Nifti1Image):
            #alignmentMap4D = alignmentMap.reshape(x,y,z,-1)
            alignmentMap4D = alignmentMap.T.reshape(x,y,z,-1)
            alignmentMap = ni.nifti1.Nifti1Image(alignmentMap4D, img[0].affine, img[0].header)
        else:
            raise TypeError(f'Input data must be type Nift1Image or Cifti2Image but {type(img[0])} found.')

        ni.save(alignmentMap, outputs['out_file'])

    def _gen_filename(self, name):
        import os
        
        if name == 'out_file':
            basename = os.path.basename(self.inputs.in_files[0])
            
            # recursively put extensions
            ext = ''
            base, this_ext = os.path.splitext(basename)
            while this_ext:
                ext = this_ext + ext
                base, this_ext = os.path.splitext(base)

            if ext == '.dtseries.nii':
                return os.path.join(os.getcwd(), 'alignment_map.dscalar.nii')
            else:
                return os.path.join(os.getcwd(), 'alignment_map' + ext)
    
    def _list_outputs(self):
        outputs = self.output_spec().get()
        if not isdefined(outputs['out_file']):
            outputs['out_file'] = self._gen_filename('out_file')
        return outputs



class iterativeDualRegressionInputSpec(BaseInterfaceInputSpec):
    in_files=List(File(),
        desc="List of 4D CIFTI or NIFTI files containing connectome or timeseries data to be aligned",
        exists=True,
        required=True)

    group_map=File(
        desc="4D CIFTI or NIFTI file containing connectome or timeseries data to be aligned to",
        exists=True,
        required=True)
        
    iterations=traits.Int(2,
        desc="Number of iterations of dual regression to run",
        usedefault=True,
        required=True)

    distortion_weights=File(None,
        desc="CIFTI or NIFTI file of distortion weights",
        exists=True,
        required=False)

    alignment_map=File(None,
        desc="CIFTI or NIFTI file of alignment weights",
        exists=True,
        required=False)
        
    zscorets=traits.Bool(False,
        usedefault=True,
        desc="Whether to estimate standardized timseries coefficients")

    zscoremap=traits.Bool(False,
        usedefault=True,
        desc="Whether to estimate standardized spatial coefficients")
        

class iterativeDualRegressionOutputSpec(TraitedSpec):
    timeseries=List(File(),
        desc="Path of output data.",
        exists=True)
        
    beta=List(File(),
        desc="Path of output data.",
        exists=True)

    tstat=List(File(),
        desc="Path of output data.",
        exists=True)
        
    zstat=List(File(),
        desc="Path of output data.",
        exists=True)



class iterativeDualRegression(BaseInterface):
    input_spec = iterativeDualRegressionInputSpec
    output_spec = iterativeDualRegressionOutputSpec
    
    
    def _run_interface(self, runtime):
        img = [ni.load(c) for c in self.inputs.in_files]
        groupMapImg = ni.load(self.inputs.group_map)
        
        if isinstance(img[0], Cifti2Image):
            # import
            data = [c.get_fdata() for c in img]
            groupMap = groupMapImg.get_fdata()
 
            n_features = groupMap.shape[1]

            # mask for consistency with volumetric analyses where we need to mask 
            # a bunch of  extracerebral data for computational tractability
            mask = np.any([np.any(d != 0, keepdims=True, axis=0) for d in data], axis=0)
            mask = np.squeeze(mask)

            if self.inputs.distortion_weights:
                # the distortion map is a measure of variation in distances between vetices
                # on a mesh surface, so it's only defined on the surface
                surface_mask = img[0].header.get_axis(1).surface_mask
                distortionMap = np.ones((1, data[0].shape[1]))
                distortionMap[:,surface_mask] = ni.load(self.inputs.distortion_weights).get_fdata()

            if self.inputs.alignment_map:
                alignmentMap = ni.load(self.inputs.alignment_map).get_fdata()
        elif isinstance(img[0], ni.nifti1.Nifti1Image):
            # import and vectorize
            data = [img[0].get_fdata()]
            x, y, z, t = data[0].shape
            data[0] = data[0].reshape(x*y*z,t).T

            n_features = x*y*z

            data = data + [c.get_fdata().reshape(x*y*z,t).T for c in img[1:]]
            groupMap = groupMapImg.get_fdata().reshape(x*y*z,-1).T

            # implicit masking. If there's been any masking during preprocessing
            # then we'll have a bunch of empty extracerebral (and possibly intracerebral)
            # space here that we can ignore for efficiency.
            mask = np.any([np.any(d != 0, keepdims=True, axis=0) for d in data], axis=0)
            mask = np.squeeze(mask)

            if self.inputs.distortion_weights:
                distortionMap = ni.load(self.inputs.distortion_weights).get_fdata().reshape(x*y*z,-1).T

            if self.inputs.alignment_map:
                alignmentMap = ni.load(self.inputs.alignment_map).get_fdata().reshape(x*y*z,-1).T
        else:
            raise TypeError(f'Only Nifti1Image Cifti2Image objects are supported, but {self.inputs.in_files[0]} is type {type(img[0])}')

        # apply masking and estimate regression weights
        data = [d[:,mask] for d in data]
        groupMap = groupMap[:,mask]

        if self.inputs.distortion_weights:
            distortionMap = distortionMap[:,mask]

            distortion = np.ones((1, sum(mask)))
            distortion[0,:sum(mask)] = distortionMap / np.mean(distortionMap)
        else:
            distortion=None

        if self.inputs.alignment_map:
            alignmentMap = alignmentMap[:,mask]

            alignment = np.ones((1,data[0].shape[1]))
            alignment[0,:alignmentMap.shape[1]] = alignmentMap / np.mean(alignmentMap)
        else:
            alignment=None

        betaICA = np.zeros((len(data), groupMap.shape[0], n_features))
        tstatICA = np.zeros((len(data), groupMap.shape[0], n_features))
        zstatICA = np.zeros((len(data), groupMap.shape[0], n_features))
        tsICA, betaICA[:, :, mask], tstatICA[:, :, mask], zstatICA[:, :, mask] = dr.iterative_dual_regression(
            data, groupMap, 
            iterations=self.inputs.iterations,
            distortionWeights=distortion,
            alignmentMap = alignment,
            zscorets = self.inputs.zscorets, 
            zscoremap = self.inputs.zscoremap
            )
        
        outputs = self._list_outputs()
        if isinstance(img[0], Cifti2Image):
            template_cifti = groupMapImg
            brain_model_axis = cifti_utils.get_brain_model_axis(template_cifti)

            betaICAMap = []
            tstatICAMap = []
            zstatICAMap = []
            for B,T,Z in zip(betaICA, tstatICA, zstatICA):
                lbls = [str(int(ind)) for ind in range(0, B.shape[0])]
                scalar_axis = ScalarAxis(name=np.array(lbls))

                hdr = Cifti2Header.from_axes((scalar_axis, brain_model_axis))

                betaICAMap.append(ni.Cifti2Image(B, header=hdr, nifti_header=template_cifti.nifti_header))
                tstatICAMap.append(ni.Cifti2Image(T, header=hdr, nifti_header=template_cifti.nifti_header))
                zstatICAMap.append(ni.Cifti2Image(Z, header=hdr, nifti_header=template_cifti.nifti_header))
        elif isinstance(img[0], ni.nifti1.Nifti1Image):
            betaICAMap = []
            tstatICAMap = []
            zstatICAMap = []
            for B,T,Z in zip(betaICA, tstatICA, zstatICA):
                b4D = B.T.reshape(x,y,z,-1)
                t4D = T.T.reshape(x,y,z,-1)
                z4D = Z.T.reshape(x,y,z,-1)

                betaICAMap.append(ni.nifti1.Nifti1Image(b4D, groupMapImg.affine, groupMapImg.header))
                tstatICAMap.append(ni.nifti1.Nifti1Image(t4D, groupMapImg.affine, groupMapImg.header))
                zstatICAMap.append(ni.nifti1.Nifti1Image(z4D, groupMapImg.affine, groupMapImg.header))
        else:
            raise TypeError(f'Input data must be type Nifti1Image or Cifti2Image but {type(img[0])} found.')

        for i,(ts,b,t,z) in enumerate(zip(tsICA, betaICAMap, tstatICAMap, zstatICAMap)):
            np.savetxt(outputs['timeseries'][i], ts, delimiter=",", fmt="%0.7f")
            ni.save(b, outputs['beta'][i])
            ni.save(t, outputs['tstat'][i])
            ni.save(z, outputs['zstat'][i])


    def _gen_filename(self, name):
        import os
        
        if name == 'beta' or name == 'tstat' or name =='zstat':
            basename = os.path.basename(self.inputs.in_files[0])
            
            # recursively put extensions
            ext = ''
            base, this_ext = os.path.splitext(basename)
            while this_ext:
                ext = this_ext + ext
                base, this_ext = os.path.splitext(base)
            
            if ext == '.dtseries.nii':
                name_list = [os.path.join(os.getcwd(), name + f'_{i}.dscalar.nii') 
                                for i in range(len(self.inputs.in_files))]
            else:
                name_list = [os.path.join(os.getcwd(), name + f'_{i}' + ext) 
                                for i in range(len(self.inputs.in_files))]
            return name_list

        elif name == 'timeseries':
            name_list = [os.path.join(os.getcwd(), f'timeseries_{i}.csv')
                            for i in range(len(self.inputs.in_files))]
            return name_list
    
    def _list_outputs(self):
        outputs = self.output_spec().get()
        for item in outputs.keys():
            if not isdefined(outputs[item]):
                outputs[item] = self._gen_filename(item)
        return outputs