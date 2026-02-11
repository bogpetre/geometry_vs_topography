import os
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import balanced_accuracy_score
from sklearn.neighbors import NearestCentroid

from nipype.interfaces.base import (
    BaseInterface,
    BaseInterfaceInputSpec,
    traits,
    File,
    Str,
    TraitedSpec,
    isdefined
)
from traits.api import (
    List,
    Bool
)

from ..hcp_utils_ext.utils import load_parcellation
from ..nearestcentroidclf.model import extract_region_data, get_img_data, get_clf_mat


class NearestCentroidClfInputSpec(BaseInterfaceInputSpec):
    cifti=File(
        desc="4D CIFTI or NIFTI files containing to-be-classified data",
        exists=True,
        required=True)

    labels=List(Str(),
        required=False,
        desc="List of true labels. Order must match volumes of cifti. If not specified cifti \
              volume labels are used instead.")

    groupkfold=Bool(True,
        usedefault=True,
        desc="Do group k-fold. Folds will not fragment groups.")

    groups=List(Str(),
        required=False,
        default=None,
        usedefault=True,
        desc="Groups to use for k-Fold CV. If not specified LR/RL strings from cifti volume \
              labels are used.")

    atlas=File(required=True,
        desc="cifti dlabels file to use to define parcels. Each parcel is evaluated \
             independently and outputted as a separate row of the output csv file.")


class NearestCentroidClfOutputSpec(TraitedSpec):
    clf_file=File(
        desc="Output clf values.",
        exists=True)

    label_file=File(
        desc="Output labels.",
        exists=True)


class NearestCentroidClf(BaseInterface):
    input_spec=NearestCentroidClfInputSpec
    output_spec=NearestCentroidClfOutputSpec

    def _run_interface(self, runtime):
        parcellation = load_parcellation(self.inputs.atlas)

        pipeline = Pipeline([
            ('nearestcentroid', NearestCentroid()) 
        ])


        X, y, groups = get_img_data(self.inputs.cifti)
        if self.inputs.groupkfold:
            if self.inputs.groups:
                groups = self.inputs.groups
            crossValidator = StratifiedGroupKFold(n_splits=2)
        else:
            groups = None
            crossValidator = StratifiedKFold(n_splits=2)

        if self.inputs.labels:
            y = self.inputs.labels

        scores, y_set = get_clf_mat(pipeline, crossValidator, balanced_accuracy_score, 
            X, y, groups=groups,
            parcellation=parcellation, region_ind=np.arange(len(parcellation.labels)))


        df = pd.DataFrame.from_dict(scores, orient='index')

        outputs = self._list_outputs()

        df.to_csv(outputs['clf_file'], index_label='label')
        np.savetxt(outputs['label_file'], y_set, fmt='%s', delimiter=',')

    def _gen_filename(self, name):
        import os

        if name == 'clf_file':
            return os.path.join(os.getcwd(), 'binary_clf_performance.csv')
        elif name == 'label_file':
            return os.path.join(os.getcwd(), 'binary_clf_labels.csv')

    def _list_outputs(self):
        outputs = self.output_spec().get()
        if not isdefined(outputs['clf_file']):
            outputs['clf_file'] = self._gen_filename('clf_file')
        if not isdefined(outputs['label_file']):
            outputs['label_file'] = self._gen_filename('label_file')
        return outputs
