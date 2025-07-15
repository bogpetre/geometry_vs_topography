import os
import sys
import argparse
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import balanced_accuracy_score
from sklearn.neighbors import NearestCentroid

from ..hcp_utils_ext.utils import load_parcellation
from .model import extract_region_data, get_img_data, get_clf_mat

def main():
    parser = argparse.ArgumentParser(description="NearestCentroid classification of CIFTI volumes")
    parser.add_argument('--cifti', required=True, help="4D CIFTI volume to evaluate")
    parser.add_argument('--labels', nargs='*', default=None, required=False,
                        help='List of true labels. Order must match volumes of cifti. If not specified cifti \
                            volume labels are used instead.')
    groupkfold = parser.add_argument_group('Group k-Fold CV')
    groupkfold.add_argument('--groupkfold', action='store_true', default=False,
                        help="Do group k-fold. Folds will not fragment groups.")
    groupkfold.add_argument('--groups', nargs='*', default=None, required=False,
                            help='Groups to use for k-Fold CV. If not specified LR/RL strings from cifti volume labels are used.')                            
    parser.add_argument('--atlas', type=str, required=True,
                        help='cifti dlabels file to use to define parcels. Each parcel is evaluated \
                            independently and outputted as a separate row of the output csv file.')
    parser.add_argument('--out', type=str, required=True, help='Output csv file.')

    args = parser.parse_args()

    parcellation = load_parcellation(args.atlas)

    pipeline = Pipeline([
        ('nearestcentroid', NearestCentroid()) 
    ])


    X, y, groups = get_img_data(args.cifti)
    if args.groupkfold:
        if args.groups:
            groups = args.groups
        crossValidator = StratifiedGroupKFold(n_splits=2)
    else:
        groups = None
        crossValidator = StratifiedKFold(n_splits=2)

    if args.labels:
        y = args.labels

    scores, y_set = get_clf_mat(pipeline, crossValidator, balanced_accuracy_score, 
        X, y, groups=groups,
        parcellation=parcellation, region_ind=np.arange(len(parcellation.labels)))


    df = pd.DataFrame.from_dict(scores, orient='index')
    df.to_csv(args.out, index_label='label')
