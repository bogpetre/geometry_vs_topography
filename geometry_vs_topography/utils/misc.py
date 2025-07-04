import numpy as np

def rois_to_data_ind(rois, roi_nodes, data_nodes) -> list:
    '''
    Converts a set of masks defined by unique array values in <rois> into a list of 
    indices into data by matching correspoding roi_nodes to indices of data_nodes.
    
    Script assumes data and rois are in the same 'space', meaning that values in 
    data_nodes and roi_nodes can be interpretted equivalently, that a particular value
    in one maps to the same location in physical space as the same value in the other.
    This is not enforced right now, so the user needs to be careful to manually enforce 
    this.
    '''
    
    assert rois.shape[-1] == len(roi_nodes)
    
    rois = reshape_X_to_data(rois, roi_nodes, data_nodes)

    uniq_rois = np.unique(np.round(rois))
    ind = []
    for roi in uniq_rois:
        if roi != 0:
            this_ind = np.where(roi == rois)[-1]
            ind.append(this_ind)
    
    return ind
