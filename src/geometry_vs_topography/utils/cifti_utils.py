#!/usr/bin/env python
# coding: utf-8

# cifti_utils is intended to store function that operate directly on cifti files. This file is meant
# to mask the nibabel cifti API calls, which are esoteric, and replace them with more intitively named
# functionality

import numpy as np
import nibabel as ni
from nibabel.cifti2.cifti2 import Cifti2Image, Cifti2Label
from nibabel.gifti.gifti import GiftiImage, GiftiLabel
from nibabel.nifti1 import Nifti1Image
from nibabel.nifti2 import Nifti2Image
from warnings import warn
from typing import Union # replace with | in place of Union[type1, type2] in python 3.10 and up


def _get_cifti(cifti: Union[Cifti2Image, Cifti2Label], 
                   ctx_surf_left=None, ctx_surf_right=None, ref=None) -> list:
    '''
    A monolithic function that returns everything we ever need from a cifti file. Wrappers
    pull specific subsets of items as needed. Hopefully in the future this can be made more
    modular by substituting new code for the wrapper and retiring this function.
    
    Input :: 
        cifti - nibabel cifti object, obtained using ni.load()
        surf_left - nibabel left surface object, obtained using ni.load()
        surf_right - nibabel right surface object, obtained using ni.load()
        ref - nibabel cifti object, obtained using ni.load(), that should determine the indexing of
            returned data
        
        surfaces are matched to vertices in cifti based on brain_model specified in cifti header.
        CIFTI_STRUCTURE_CORTEX_LEFT and CIFTI_STRUCTURE_CORTEX_RIGHT are mapped to ref_left and 
        ref_right, respectively.
        
    Output ::
        data           - array of scalar values corresponding to each vertex/voxel in ref but containing 
                         data from the cifti
        node_indices   - indices of data. If vertices are empty they aren't included in data, but empty
                         vertices are nevertheless tracked and node_indices are incremented accordingly
        structure_ind  - indexed array indicating unique structures
        structure_lbl  - labels of unique structures. structure_ind == x corresponds to structure_lbl[x]
        surface        - boolean array indicate whether or not the corresponding point in data is a 
                         surface vertex
        vertex         - surface vertex indices corresponding to node_indices. By combining structure_ind,
                         structure_lbl, and vertex you can get the corresponding index of the original 
                         surface mesh back. These will perfectly correlate with node_indices +/- some offset
                         depending on the order in which surfaces are entered into the cifti file.
        
    Major Bug: nodes that returned for surfaces are indexed by the reference volumes, but nodes that
     are returned for the subcortex have no reference. These nodes are not guaranteed to correspond
     across invocations of this function on different cifti files!
     
    This isn't a very robust function. It assumes surface indices come first before volume indices in
    the data matrix, it assumes that nodes from each hemisphere are listed consecutively and that
    volume indices are also consecutive without fragmentation and mixing with surfaces. This is true
    for fsLR/HCP91k data but I as far as I know it's not enforced in any way by the CIFTI standard.
    '''
    
    if not ref:
        ref = cifti

    
    # get slice objects for each structure
    cifti_bm = get_brain_model_axis(cifti)
    ref_bm = get_brain_model_axis(ref)
    
    if np.any(ref_bm.affine) or np.any(cifti_bm.affine):
        assert all(ref_bm.affine.reshape(-1) == cifti_bm.affine.reshape(-1)), 'ref is not aligned with cifti'
    
    cifti_structs = [struct for struct in cifti_bm.iter_structures()]
    ref_structs = [struct for struct in ref_bm.iter_structures()]
    
    
    
    # get node indices correspondint to structs
    if 'get_header' in dir(cifti):
        cifti_hdr = cifti.get_header()
    elif 'header' in dir(cifti):
        cifti_hdr = cifti.header
    else:
        raise ValueException('Cifti2Image does not have expected header query functions. This is probably due to a version mismatch with what was present during this functions design')

    cifti_index_map = [cifti_hdr.get_index_map(i)
                       for i in range(cifti_hdr.number_of_mapped_indices)
                       if cifti_hdr.get_index_map(i).indices_map_to_data_type == 'CIFTI_INDEX_TYPE_BRAIN_MODELS']
    cifti_index_map_bm = [brain_models 
                          for index_map in cifti_index_map
                          for brain_models in index_map.brain_models]


    assert len(cifti_index_map_bm) == len(cifti_structs)
    for structs, indices in zip(cifti_structs, cifti_index_map_bm):
        assert structs[0] == indices.brain_structure, 'ref structures and index maps are misaligned, this is very strange'
        
        
    if 'get_header' in dir(ref):
        ref_hdr = ref.get_header()
    elif 'header' in dir(ref):
        ref_hdr = ref.header
    else:
        raise ValueException('Cifti2Image does not have expected header query functions. This is probably due to a version mismatch with what was present during this functions design')

    ref_index_map = [ref_hdr.get_index_map(i)
                     for i in range(ref_hdr.number_of_mapped_indices)
                     if ref_hdr.get_index_map(i).indices_map_to_data_type == 'CIFTI_INDEX_TYPE_BRAIN_MODELS']
    ref_index_map_bm = [brain_models 
                        for index_map in ref_index_map
                        for brain_models in index_map.brain_models]

    assert len(ref_index_map_bm) == len(ref_structs)
    for structs, indices in zip(ref_structs, ref_index_map_bm):
        assert structs[0] == indices.brain_structure, 'ref structures and index maps are misaligned, this is very strange'
    
    
    if np.any(ref_bm.affine):
        cifti_ijk_to_xyz = cifti_index_map[0].volume.transformation_matrix_voxel_indices_ijk_to_xyz.matrix    
        ref_ijk_to_xyz = ref_index_map[0].volume.transformation_matrix_voxel_indices_ijk_to_xyz.matrix
    
    # construct returned arrays based on ref image order
    data = []
    node_indices = []
    structure_ind = [] # index map indicating unique structures
    structure_lbl = [] # CIFTI name of structure
    surface = [] # boolean indicating if surface
    vertex = [] # index into corresponding surface mesh

    cifti_data = cifti.get_fdata()
    ref_data = ref.get_fdata()

    max_node_index = -1
    max_structure_ind = -1

    for structs, indices in zip(ref_structs, ref_index_map_bm):
        this_cifti_struct = [s for s in cifti_structs if s[0] == structs[0]]
        these_cifti_indices = [i for i in cifti_index_map_bm if i.brain_structure == indices.brain_structure]

        assert len(this_cifti_struct) == len(these_cifti_indices), 'This should be guaranteed by prior assertions'

        if len(this_cifti_struct) == 1:
            this_cifti_struct = this_cifti_struct[0]
            these_cifti_indices = these_cifti_indices[0]


            ref_slices = structs[1]
            cifti_slices = this_cifti_struct[1]
            this_ref_data = np.full((cifti_data.shape[0], ref_data[:,ref_slices].shape[1]), np.nan)
            this_cifti_data = cifti_data[:, cifti_slices]


            if all(structs[2].surface_mask):
                # if surface

                # find intersection of ref and cifti
                cifti_indices = [v for v in these_cifti_indices.vertex_indices]
                ref_indices = [v for v in indices.vertex_indices]

                # this gives us this equivalence mapping:
                # cifti[cifti_subset_ind] <-> ref[ref_subset_ind]
                cifti_subset_ind = [i for i,v in enumerate(cifti_indices) if v in ref_indices]
                ref_subset_ind = [i for i,v in enumerate(ref_indices) if v in cifti_indices]

                this_ref_data[:,ref_subset_ind] = this_cifti_data[:,cifti_subset_ind]

                # append data to returned variables
                data.append(this_ref_data)
                node_indices.append([v + max_node_index + 1
                                     for v in indices.vertex_indices])
                structure_ind.append(max_structure_ind + np.ones(len(node_indices[-1])))
                structure_lbl.append(structs[0])
                surface.append(structs[2].surface_mask)
                vertex.append(ref_indices)

            elif not any(structs[2].surface_mask):
                # if volume

                # find intersection of ref and cifti
                cifti_indices = [tuple(v) for v in these_cifti_indices.voxel_indices_ijk]
                ref_indices = [tuple(v) for v in indices.voxel_indices_ijk]

                # this gives us this equivalence mapping:
                # cifti[cifti_subset_ind] <-> ref[ref_subset_ind]
                cifti_subset_ind = [i for i,v in enumerate(cifti_indices) if v in ref_indices]
                ref_subset_ind = [i for i,v in enumerate(ref_indices) if v in cifti_indices]

                this_ref_data[:,ref_subset_ind] = this_cifti_data[:,cifti_subset_ind]

                # append data to returned variables
                data.append(this_ref_data)
                node_indices.append([v + max_node_index + 1
                                     for v in range(data[-1].shape[1])])
                structure_ind.append(max_structure_ind + np.ones(len(node_indices[-1])))
                structure_lbl.append(structs[0])
                surface.append(structs[2].surface_mask)
                vertex.append([np.nan]*data[-1].shape[1])

            else:
                raise NotImpelementedError(f'{structs[0]} is a mix of surface and volume indices which is not supported')

            max_node_index = max(node_indices[-1])
            max_structure_ind = max(structure_ind[-1])
        
    return np.hstack(data), np.hstack(node_indices), np.hstack(structure_ind), np.hstack(structure_lbl), np.hstack(surface), np.hstack(vertex)

def get_cifti_data(cifti: Union[Cifti2Image, Cifti2Label], 
                   ref_left=None, ref_right=None, ref=None) -> list:
    
    data, node_indices, _, _, _, _ = _get_cifti(cifti, ref_left, ref_right, ref)
    
    return data, node_indices

# used when saving connectome or preproc data to get a brain model axis
def get_brain_model_axis(cifti: Cifti2Image) -> ni.cifti2.BrainModelAxis:
    if 'get_header' in dir(cifti):
        cifti_hdr = cifti.get_header()
    elif 'header' in dir(cifti):
        cifti_hdr = cifti.header
    else:
        raise ValueException('Cifti2Image does not have expected header query functions. This is probably due to a version mismatch with what was present during this functions design')

    axes = [cifti_hdr.get_axis(i) for i in range(cifti.ndim)]

    # find brain_model_axis
    brain_model_axis = [axis for axis in axes if isinstance(axis, ni.cifti2.cifti2_axes.BrainModelAxis)]
    assert len(brain_model_axis) == 1, f'Found {len(brain_model_axis)} but expected 1'
    brain_model_axis = brain_model_axis[0]
    
    return brain_model_axis