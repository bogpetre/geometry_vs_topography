'''
this script needs an option for handling a zero patch in an image. 
Alignment can produce these around boundaries which might create the illusion of 
between subject correlations that are driven by the same areas having data moved 
out of them.

Maybe ignoring exact zeros can be added as an optional argument, or maybe it can
be used internally as a warning flag. e.g. if the dif between zero-excluded 
correlation and actual correlation is too large, throw a warning
'''

from os.path import exists

import nibabel as ni
from nibabel.cifti2 import Cifti2Header
from nibabel.cifti2.cifti2 import Cifti2Image
from nibabel.gifti.gifti import GiftiImage
from nibabel.nifti1 import Nifti1Image

from argparse import ArgumentParser
from pathlib import Path
from warnings import warn

from ..utils import image_utils, misc

import numpy as np

def init_argparse() -> ArgumentParser:
    parser = ArgumentParser(
        prog="pairwise_parcel_op",
        usage="%(prog)s <input_4d_file_1> <input_4d_file_2> <output> --surf <left_surf> <right_surf> --parcels <parcellation>",
        description="Perform a binary operation on matching parcels from two input files."
    )
    parser.add_argument(
        "-v", "--version", action="version",
        version = f"{parser.prog} version 0.1.0"
    )
    parser.add_argument('input_1',
                       help='3D CIFTI file')
    parser.add_argument('input_2',
                       help='3D CIFTI file')
    parser.add_argument('out_file', 
                       help='Path to write output to.')
    parser.add_argument('-s','--surfs', required=False, nargs=2, default=[None,None],
                       help='[left] [right] hemisphere reference gifti surface files')
    parser.add_argument('-p','--parcels', required=True,
                       help='Masks with unique values for each unique parcel. 4D if parcels overlap. With ParcelAxis if CIFTI')
    parser.add_argument('-o','--operation', required=True,
                       help='Operation to perform. Options: correlation')
    return parser


def main():
    '''
    TODO:
    - Check that SRC_SCAN and TARGET_PARCELS are in the same space
    - add support for surface and volume inputs
    - autodetect input format (grayordinate, surface or volume)
    '''

    # handle inputs
    parser = init_argparse()
    args = parser.parse_args()

    out_file = Path(args.out_file)
    
    input_1 = Path(args.input_1)
    if exists(input_1):
        if input_1 == out_file:
            print('Cannot overwrite input file. Please select a unique output')
            raise SystemExit(1)
            
        input_1 = ni.load(input_1)
    else:
        print('Could not find source scan file {0}'.format(input_1))
        raise SystemExit(1)  
        
    input_2 = Path(args.input_2)
    if exists(input_2):
        if input_2 == out_file:
            print('Cannot overwrite input file. Please select a unique output')
            raise SystemExit(1)
            
        input_2 = ni.load(input_2)
    else:
        print('Could not find source scan file {0}'.format(input_2))
        raise SystemExit(1)
    
    
    surf_left = args.surfs[0]
    surf_right = args.surfs[1]
    if isinstance(input_1, (Cifti2Image, GiftiImage)):
        if not isinstance(input_2, type(input_1)):
            raise ValueError(f'Input files are different types {type(input_1)} vs. {type(input_2)}.')
            
        surf_left = Path(surf_left)
        if exists(surf_left):
            surf_left = ni.load(surf_left)

        surf_right = Path(surf_right)
        if exists(surf_right):
            surf_right = ni.load(surf_right)
        

    if args.parcels and exists(Path(args.parcels)):
        parcels = ni.load(args.parcels)
    else:
        raise ValueError(f'Could not find parcels file {args.parcels}.')
        
    operation = args.operation
    if operation == 'correlation':
        operation = lambda x,y: np.corrcoef(x,y)[0,1]
    elif operation == 'zcorrelation':
        operation = lambda x,y: np.arctanh(np.corrcoef(x,y)[0,1])
    elif operation == 'cosine':
        operation = lambda x,y: (x/np.linalg.norm(x)).T @ (y/np.linalg.norm(y)).T
    else:
        raise ValueError('operation not valid')


    # load data
    
    if isinstance(parcels, Cifti2Image):
        if isinstance(parcels.header.get_axis(1), ni.cifti2.cifti2_axes.BrainModelAxis):
            parcel_data, parcel_nodes = image_utils.get_data(parcels, surf_left, surf_right)
        elif isinstance(parcels.header.get_axis(1), ni.cifti2.cifti2_axes.ParcelsAxis):
            raise NotImplemented('please rewrite index parcels into data to index data into parcels')
        else:
            raise TypeError('Could not autodetect format of input cifti.')
    else:
        raise TypeError('Only CIFTI files are currently supported')

    data_1, data_1_nodes = image_utils.get_data(input_1, surf_left, surf_right, parcels)
    data_2, data_2_nodes = image_utils.get_data(input_2, surf_left, surf_right, parcels)

    roi_ind = misc.rois_to_data_ind(parcel_data, parcel_nodes, parcel_nodes)
        
    if len(data_1.shape) > 1:
        results = np.zeros((data_1.shape[0], parcel_data.shape[-1]))
    else:
        results = np.zeros((1,parcel_data.shape[-1]))
        data_1 = np.array([data_1])

    if len(data_2.shape) == 1:
        data_2 = np.array([data_2])
       
    for i,ind in enumerate(roi_ind):
        for row_ind in range(data_1.shape[0]):
            finite = np.isfinite(np.squeeze(data_1[row_ind,ind])) & np.isfinite(np.squeeze(data_2[row_ind,ind]))
            finite_ind = ind[finite]
            n_bad_ind = len(ind) - len(finite_ind)
            if n_bad_ind > 0:
                warn('{0}/{1} non-finite (NaN or Inf) values for parcel 1 omitted'.format(n_bad_ind,len(ind)))

            results[row_ind, ind] = operation(np.squeeze(data_1[row_ind, finite_ind]), 
                                              np.squeeze(data_2[row_ind, finite_ind]))
    
    kwargs = {}
    if isinstance(parcels, Cifti2Image):
        new_header = Cifti2Header.from_axes((input_1.header.get_axis(0), parcels.header.get_axis(1)))
        results = Cifti2Image(dataobj=results,
                              header=new_header, nifti_header=parcels.nifti_header)
    else:
        raise TypeError(f'Input data type {type(input_1)} is not supported')

    ni.save(results, out_file, **kwargs)
