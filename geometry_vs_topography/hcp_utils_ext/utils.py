import sys

import numpy as np
import nibabel as ni
import sklearn as sk
from warnings import warn

package_directory = '/dartfs-hpc/rc/home/m/f0042vm/software/hcp_utils'
if package_directory not in sys.path:
    sys.path.insert(0, package_directory)
import hcp_utils as hcp

def uncortex_data(data):
    '''
    Removes medial wall from data to conform to HCP format. Inverse of hcp.cortex_data()
    '''

    hcp_data = np.hstack([data[:,hcp.vertex_info['grayl']], 
                          data[:,hcp.vertex_info['grayr']+hcp.vertex_info['num_meshl']],
                          data[:,64984:]])
    return hcp_data

def load_parcellation(path):
    '''
    Creates an HCP compliant parcellation (type: scikit-learn Bunch obj) from an indexed HCP style CIFTI file
    '''

    atlas = ni.load(path)
    atlas_data = atlas.get_fdata()
    if any(s > 91282 for s in atlas_data.shape):
        warn('Size of image is greater than 91282. Attempting medial wall removal')
        hcp_atlas_data = uncortex_data(atlas_data)[0]
    else:
        hcp_atlas_data = atlas_data[0]

    assert(len(hcp_atlas_data) == 91282, 'Size of data must be 91282 for HCP compliance')

    ids = set(hcp_atlas_data)
    nontrivial_ids = set(hcp_atlas_data)
    nontrivial_ids.remove(0)
    
    map_all = hcp_atlas_data
    label_table = atlas.header.get_axis(0).label[0]

    labels = {i: label_table[key][0] for i,key in enumerate(label_table)}
    rgba = {key: np.array(label_table[key][1]) for key in label_table}

    return sk.utils.Bunch(ids=np.array(list(ids), dtype=int), 
                          map_all=map_all,
                          labels=labels,
                          rgba=rgba, 
                          nontrivial_ids=np.array(list(nontrivial_ids), dtype=int))

def mk_lbl_map(labels_old, labels_new):
    '''
    Creates a dictionary of keys where each key is a new label and for each 
    key the dictionary returns finer scaled old labels. Also creates an inverse
    map that for each old label returns the new label.
    labels_old - 1xn list of old (fine) labels. Should be unique.
    labels_new - 1xn list of new (coarser) labels. Can be nonunique.
    '''

    # gets unique elements of X without changing their order
    def _uniq_stable(X):
        X_new = []
        for x in X:
            if x not in X_new:
                X_new.append(x)
        return X_new
    
    label_map = dict.fromkeys(_uniq_stable(labels_new), [])
    inverse_label_map = dict.fromkeys(_uniq_stable(labels_old),[])
    
    for lbl_old, lbl_new in zip(labels_old,labels_new):
        label_map[lbl_new] = label_map[lbl_new] + [lbl_old]
        inverse_label_map[lbl_old] = inverse_label_map[lbl_old] + [lbl_new]

    return label_map, inverse_label_map

def downsample_parcellation(parcellation, interlabel_map, inv_interlabel_map):
    '''
    parcellation - sklearn Bunch object containing an hcp_utils style parcellation. See hcp.yeo7 for an example
    labels_old - a dictionary of new labels, containing list of old labels for each element.

    returns a new hcp parcellation that uses the downsampled labels
    '''

    ids = np.arange(len(interlabel_map.keys())+1, dtype=int)

    labels = {i+1: label for i,label in enumerate(interlabel_map)}
    labels[0] = ''
    inv_labels = {label: i+1 for i,label in enumerate(interlabel_map)}

    inv_labels_old = {v: k for k,v in parcellation.labels.items()}

    nontrivial_ids = np.arange(len(interlabel_map.keys()), dtype=int) + 1

    map_all = []
    rgba = dict.fromkeys(range(len(labels)))
    rgba[0] = np.array([1,1,1,1], dtype=float)
    for i in parcellation.map_all:
        if i == 0:
            map_all.append(0)
        else:
            old_label = parcellation.labels[i] # a label from the fine (eg Ctx_V1_L)
            new_label = inv_interlabel_map[old_label][0] # a label from the coarse atlas (eg visual_early_L)
            new_indx = inv_labels[new_label] # a number indexing a label in the coarse atlas (eg 1)
            map_all.append(new_indx)
    
            example_fine_label = interlabel_map[new_label][0] # the first fine label subsummed by the new coarse label
            old_indx = inv_labels_old[example_fine_label]
            new_rgba = parcellation.rgba[old_indx]
            rgba[new_indx] = new_rgba
        
    return sk.utils.Bunch(ids=ids,
                          map_all=np.array(map_all),
                          labels=labels,
                          rgba=rgba, 
                          nontrivial_ids=nontrivial_ids)
