# generic nibabel image functions which behave identically regardless of whether arguments
# are nifti, cifti or gifti objects

import numpy as np
import nibabel as ni

from . import cifti_utils

def get_data(ni_img, surf_left=None, surf_right=None, ref_img=None):
    '''
    Generic data function for grabbing vectorized image data. Should work on Cifti, 
    Nifti and Gifti nibabel types. Behavior of keyword arguments will differ for each type.
    '''
    
    if ref_img and not type(ni_img) is type(ref_img):
        raise TypeError(f'input type {type(ni_img)} does not match reference img type {type(ref_img)}')
    
    if isinstance(ni_img, ni.cifti2.cifti2.Cifti2Image):
        data, data_nodes = cifti_utils.get_cifti_data(ni_img, surf_left, surf_right, ref_img)
    else:
        raise TypeError(f'Unrecognized input data type {type(ni_img)}. Only CIFTI files are currently supported')
                    
    return data, data_nodes

def get_TR(ni_img):
    '''
    Generic function to return TR of Cifti, Nifti or Gifti files
    '''
    
    if isinstance(ni_img, ni.cifti2.cifti2.Cifti2Image):
        tr = cifti_utils.get_cifti_TR(ni_img)
    elif isinstance(ni_img, ni.nifti1.Nifti1Image):
        tr = ni_img.header.get_zooms()[3]
    elif isinstance(ni_img, ni.nifti2.Nifti2Image):
        raise NotImplementedError(f'Input type nibabel.nifti2.Nifti2Image is not yet supported.');
        '''
        Implementing support for Nifti2Images should be straightforward, but requires testing.
        In particular, make sure nibabel.nifti2.Nifti2Image.header.get_best_affine() makes sense.
        '''
    elif isinstance(ni_img, ni.gifti.gifti.GiftiImage):
        raise TypeError('Gifti files do not have TR information. You must supply this manually')
    else:
        raise TypeError(f'Unrecognized input data type {type(ni_img)}')
                    
    return tr
    
