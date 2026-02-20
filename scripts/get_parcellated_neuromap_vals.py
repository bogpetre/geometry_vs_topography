import os, sys
import nibabel as ni
import numpy as np
import hcp_utils
import json
import argparse
from neuromaps import datasets, nulls, images, parcellate, transforms
from importlib_resources import files
from joblib import Parallel, delayed

#CONFIG_PATH = files('geometry_vs_topography.etc').joinpath('config.json')
#with open(CONFIG_PATH) as f:
#    config = json.load(f)

NEUROMAPS = [
    ('abagen', 'genepc1', 'fsaverage', '10k'),
    ('hcps1200', 'myelinmap', 'fsLR', '32k'),
    ('hcps1200', 'thickness', 'fsLR', '32k'),
    ('hill2010', 'devexp', 'fsLR', '164k'),
    ('hill2010', 'evoexp', 'fsLR', '164k'),
    ('margulies2016', 'fcgradient01', 'fsLR', '32k'),
    ('neurosynth', 'cogpc1', 'MNI152', '2mm'),
    ('raichle', 'cbf', 'fsLR', '164k'),
    ('reardon2018', 'scalinghcp', 'civet', '41k'),
    ('satterthwaite2014', 'meancbf', 'MNI152', '1mm'),
    ('xu2020', 'FChomology', 'fsLR', '32k'),
    ('xu2020', 'evoexp', 'fsLR', '32k')]
    
    
surfL = ni.load(files('hcp_utils').joinpath('data/S1200.L.sphere.32k_fs_LR.surf.gii'))
NVERTSL = surfL.agg_data()[0].shape[0]

surfR = ni.load(files('hcp_utils').joinpath('data/S1200.R.sphere.32k_fs_LR.surf.gii'))
NVERTSR = surfR.agg_data()[0].shape[0]
    
def run_single_map(author, model, space, res, nperm, seed, parcellater, outdir):
    annot = datasets.fetch_annotation(source=author, desc=model)

    if 'hill2010' in author:
        annot = transforms.fslr_to_fslr(annot, '32k', hemi='R')
        full_data = np.full(NVERTSL + NVERTSR, np.nan)
        full_data[NVERTSL:] = annot[0].agg_data()
        annot = full_data
    elif space == 'fsLR':
        # in case data is in 164k rather than 32k space
        annot = transforms.fslr_to_fslr(annot, '32k')
    elif 'fsaverage' in space:
        annot = transforms.fsaverage_to_fslr(annot, '32k')
    elif 'MNI152' in space:
        annot = transforms.mni152_to_fslr(annot, '32k')
    elif 'civet' in space:
        annot = transforms.civet_to_fslr(annot, '32k')
    
    if nperm:
        annot = nulls.alexander_bloch(annot, n_perm=nperm, seed=seed, atlas='fsLR', density='32k')
        annot_regions = np.array([parcellater.transform(map, 'fsLR') for map in annot.T]).T

        # if the medial wall completely subsumes a parcel then its value will be 0.0. We need to make these nans
        # so that we can ommit them from permutation tests
        annot_regions[annot_regions == 0.0] = np.nan
    else:
        annot_regions = parcellater.transform(annot, 'fsLR')
    
    np.savetxt(f'{outdir}/{author}_{model}.csv', annot_regions, delimiter=',')
    

def main():
    parser = argparse.ArgumentParser(description="Get spatial models of 12 neuromaps aligned with the sensory association axis")
    parser.add_argument('--out', type=str, required=True,
                        help='Path to output directory')
#    parser.add_argument('--atlas', type=str, required=False, default=config['canlab2024']['path'],
    parser.add_argument('--atlas', type=str, required=True, 
                        help='cifti dlabels file to use to define parcells. Should be in fsLR 32k space, matching the HCP data')
    parser.add_argument('--nperms', type=int, required=False, default=None,
                        help='Number of spatial permutations to compute. If unset will return unpermuted map')
    parser.add_argument('--seed', type=int, required=False, default=5900,
                        help='Seed for the random number generator for reproducible results')
    parser.add_argument('--cpus', type=int, required=False, default=-1,
                        help='Number of CPUs to use when parallelizing. Reduce number of CPUs if you do not have enough memory.')
    
    args = parser.parse_args()
    
    parcellation = images.dlabel_to_gifti(args.atlas)
    parcellation = images.relabel_gifti(parcellation)
    parcellater = parcellate.Parcellater(parcellation, 'fslr').fit()

    n_perm=args.nperms
    seed=args.seed

    Parallel(n_jobs=args.cpus)(
        delayed(run_single_map)(author, model, space, res, n_perm, seed, parcellater, args.out)
        for author, model, space, res in NEUROMAPS
    )

        
if __name__ == "__main__":
    main()

