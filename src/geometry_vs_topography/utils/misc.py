import numpy as np

def reshape_X_to_data(X, X_nodes, data_nodes) -> np.ndarray:
    # map to data space
    # 1) mask with data
    ind = np.array([i for i,s in enumerate(X_nodes) if s in data_nodes])
    X = X[...,ind]
    X_nodes = X_nodes[ind]
    # 2) expand to data space missing from roi space
    dn_to_Xn = np.vstack([np.array([i, np.where(X_nodes == n)[0][0]], dtype=int)
                             for i,n in enumerate(data_nodes) if n in X_nodes])
    if len(X.shape) > 1:
        X_data = np.zeros((X.shape[0], len(data_nodes)))
    else:
        X_data = np.zeros(len(data_nodes))
    X_data[..., dn_to_Xn[...,0]] = X[...,dn_to_Xn[...,1]]
    
    return X_data

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
