import os
import sys
import argparse
import nibabel as ni
import numpy as np
import pandas as pd
import re

from sklearn.utils import Bunch
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import balanced_accuracy_score
from sklearn.neighbors import NearestCentroid


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

    assert (len(hcp_atlas_data) == 91282), 'Size of data must be 91282 for HCP compliance'

    ids = set(hcp_atlas_data)
    nontrivial_ids = set(hcp_atlas_data)
    nontrivial_ids.remove(0)
    
    map_all = hcp_atlas_data
    label_table = atlas.header.get_axis(0).label[0]

    labels = {i: label_table[key][0] for i,key in enumerate(label_table)}
    rgba = {key: np.array(label_table[key][1]) for key in label_table}

    return Bunch(ids=np.array(list(ids), dtype=int), 
                          map_all=map_all,
                          labels=labels,
                          rgba=rgba, 
                          nontrivial_ids=np.array(list(nontrivial_ids), dtype=int))

def extract_region_data(hcp91k_data, mask):
    # mask data  
    X = hcp91k_data[:,mask]

    # replace nans
    X = pd.DataFrame(X)
    X = X.apply(lambda row: row.fillna(row.mean()), axis=1)
    X = np.array(X)

    return X

def get_img_data(imgpath):
    
    img = ni.load(imgpath)
    X = img.get_fdata()

    y = [re.sub(r'Task-([A-Z]+?)-([a-zA-Z0-9\-]+?)-[LR].*',r'\1_\2',x)
            for x in img.header.get_axis(0).name]

    direction = [re.sub(r'.*-([LR]+?)-.*',r'\1',x)
                    for x in img.header.get_axis(0).name]

    return X, y, direction

def get_clf_mat(model, cv, scorer,
    img, y_label, groups=None, parcellation=None, region_ind=None):
    '''
    Returns confusion matrix for the pairwise discriminability of every combination
    of unique contrast condition from subject using the model specified by 'model'
    and cross validation fold generator specified by cv. Classification performance
    is measured by scorer. This procedure is repeated for each region from
    'parcellation' specified by region_ind. The lower triangle of the confusion 
    matrix is returned.
    '''

    # extract region and evaluate model
    scores = {parcellation.labels[key]: [] for key in parcellation.labels}
    y_set = sorted(list(dict.fromkeys(y_label)))

    for ind in region_ind:
        this_label = parcellation.labels[ind]
        mask = parcellation.map_all == ind
        X = extract_region_data(img, mask)

        if np.all(np.isnan(X)):
            continue

        for i,cond1 in enumerate(y_set):
            for cond2 in y_set[(i+1):]:
                cond_ind = np.array([j for j,lbl in enumerate(y_label) if cond1 == lbl or cond2 == lbl], dtype=int)
                this_X = X[cond_ind]

                le = LabelEncoder()
                this_y = le.fit_transform([y_label[j] for j in cond_ind])

                if groups is None:
                    this_score = cross_val_score(model, this_X, this_y, cv=cv)
                else:
                    these_groups = [groups[j] for j in cond_ind]
                    this_score = cross_val_score(model, this_X, this_y, cv=cv, groups=these_groups)

                scores[this_label].append(np.mean(this_score))

    return scores, y_set

