# This script estimates,
# - the classification performance of TDANNs and ResNets, 
# - their label agreements within dyad
# - their coarse semantic agreement on supersets
#
# These values are printed to stdout, but also saved to
# <repo_root>/derivatives/models/
#
# Note, this script must be run from a conda environment in which the TDANN repo has been installed.
# This differs from the conda environment used for the rest of this project.


from __future__ import annotations
import os
import json
from pathlib import Path
import warnings

import numpy as np
import torch

import torchvision.transforms as transforms
import torchvision.datasets as datasets

from torch.utils.data import TensorDataset, DataLoader

import time

from functools import lru_cache
from collections import Counter

import nltk

# update accordingly
file_path = Path(os.path.dirname(os.path.realpath(__file__)))
repo_root = file_path / '..'

config_path = repo_root / 'config.json'

with open(config_path) as f:
    config = json.load(f)

analysisRoot = repo_root / '..'

ST_BASE_FS = repo_root / 'resources' / 'ann_data'
os.environ['ST_BASE_FS'] = str(ST_BASE_FS)

TDANN_path = Path(config['TDANN_repo_path'])

import sys
sys.path.append(str(TDANN_path / 'demo')) # we reuse some code from here for loading old model weights

os.chdir(str(TDANN_path / 'spacetorch')) # this is the main repo code, which we need for loading old model weights

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if DEVICE == 'cpu':
    warnings.warn('GPU could not be autodetected. Please verify pytorch is compatible with your GPU. This is going to be interminably slow running on your CPU')
else:
    print('Successfully found GPU')

# configure data loader
imgnet_train_mean = [0.485, 0.456, 0.406]
imgnet_train_std = [0.229, 0.224, 0.225]

imgnet_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=imgnet_train_mean,
        std=imgnet_train_std),
])

dataset = datasets.ImageNet(root=config['imagenet_path'], split='val', transform=imgnet_transforms)

# note: we shuffle the test data, but we use the same seed below for evaluating each network instance, 
# so they draw the same test data in the same order
dataloader = DataLoader(dataset, batch_size=50, num_workers=8, pin_memory=True, shuffle=True)

# load models from checkpoint

from pathlib import Path
from vissl.models import build_model
from omegaconf import OmegaConf

from src.model import load_model_from_checkpoint

import spacetorch.utils.vissl.registration  # noqa
from spacetorch.utils.vissl.hooks import spatial_hook_generator

seed = 3

def load_model_from_checkpoint(seed, spatial='Spatial'):
    def flatten_vissl_model_dict(model_blob):
        """
        model_blob = ckpt["classy_state_dict"]["base_model"]["model"]
        Returns a flat state_dict suitable for model.load_state_dict().
        """
        sd = {}

        # trunk: keys like "base_model.conv1.weight"
        if "trunk" in model_blob:
            for k, v in model_blob["trunk"].items():
                sd[f"trunk.{k}"] = v

        # heads can be:
        #  (A) list: [ {..head0..}, {..head1..} ]
        #  (B) dict-of-heads: {"0": {...}, "1": {...}} or {0: {...}, 1: {...}}
        #  (C) flat dict: {"0.channel_bn.weight": ..., "0.clf...": ...}
        if "heads" in model_blob:
            heads = model_blob["heads"]

            if isinstance(heads, list):
                for i, head_sd in enumerate(heads):
                    for k, v in head_sd.items():
                        sd[f"heads.{i}.{k}"] = v

            elif isinstance(heads, dict):
                # detect flat style (C): keys contain dots like "0.xxx"
                sample_key = next(iter(heads.keys())) if heads else None
                if sample_key is not None and isinstance(sample_key, str) and "." in sample_key:
                    # already flat, just prefix with "heads."
                    for k, v in heads.items():
                        sd[f"heads.{k}"] = v
                else:
                    # dict-of-heads (B)
                    items = sorted(heads.items(), key=lambda kv: int(kv[0]))
                    for i_str, head_sd in items:
                        i = int(i_str)
                        for k, v in head_sd.items():
                            sd[f"heads.{i}.{k}"] = v
            else:
                raise TypeError(f"Unsupported heads type: {type(heads)}")

        return sd

    config_path = Path(repo_root) / 'derivatives' / 'models' / 'checkpoints'
    spatial = 'spatial' if spatial == 'Spatial' else 'nonspatial'
    cfg = OmegaConf.load(config_path   / 'linear_eval' / f'simclr_{spatial}_resnet18_swappedon_SineGrating2019_isoswap_3_seed_{seed}_linear_eval' / 'train_config.yaml')
    cfg['CHECKPOINT']['DIR'] = config_path
    cfg['MODEL']['WEIGHTS_INIT']['PARAMS_FILE'] = config_path  / f'simclr_{spatial}_resnet18_swappedon_SineGrating2019_isoswap_3_seed_{seed}'
    model = build_model(cfg.MODEL, cfg.OPTIMIZER)

    subdir = f"simclr_{spatial}_resnet18_swappedon_SineGrating2019_isoswap_3_seed_{seed}_linear_eval"
    CKPT_PATH = config_path / "linear_eval" / subdir / "model_final_checkpoint_phase27.torch"
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)

    blob = ckpt["classy_state_dict"]["base_model"]["model"]
    sd = flatten_vissl_model_dict(blob)

    missing, unexpected = model.load_state_dict(sd, strict=False)
    print("missing:", len(missing))
    print("unexpected:", len(unexpected))
    print("example missing:", missing[:10])
    print("example unexpected:", unexpected[:10])

    model.eval()

    return model

tdann = [load_model_from_checkpoint(s) for s in range(14)]
resnet = [load_model_from_checkpoint(s, spatial='Nonspatial') for s in range(14)]

# load imagenet labels

with open(Path(repo_root) / 'resources' / 'imagenet_labels_full.json', 'r') as file:
    json_data = json.load(file)
labels = {int(idx):content['label'] for idx, content in json_data.items()}
wnids = {int(idx):content['id'] for idx, content in json_data.items()}

# Classification labels for Figures 1A and B examples

print('Identities of stimuli from Figure 1A and 1B:')
torch.manual_seed(0)
xb, yb = next(iter(dataloader))

resnet_stim = [10,17,20]
tdann_stim = [5,19,47]

stim = tdann_stim
out_str = f'True: ' + ' '.join([f'{labels[int(p)]};' for p in yb[stim]])
print(out_str)
with torch.no_grad():
    for i,m in enumerate(tdann):
        out = m(xb[stim].reshape(len(stim),3,224,224))

        pred = out[0].softmax(dim=1).argmax(dim=1)
        out_str = f'Pred: {i}: ' + ' '.join([f'{labels[int(p)]};' for p in pred])
        print(out_str)


# Evaluate all models on ImageNet validation data
t0 = time.time()
tdann_pred = []
for i,m in enumerate(tdann):
    t1 = time.time()
    print(f'{t1-t0}: Evaluationg mdl {i}')

    m.to(DEVICE).eval()
    torch.manual_seed(0)
    mdl_pred = []
    for j,(xb, _) in enumerate(dataloader):
        xb = xb.to(DEVICE, non_blocking=True)
        with torch.no_grad():
            this_pred = m(xb)[0].softmax(dim=1).argmax(dim=1)

        mdl_pred.append(this_pred.to('cpu'))
    tdann_pred.append(torch.cat(mdl_pred))

# Stack into (M, N) then transpose to (N, M)
tdann_pred_matrix = torch.stack(tdann_pred, dim=0).T   # shape: (N_stim, N_models)

# Move to CPU and NumPy
pred_np = tdann_pred_matrix.cpu().numpy()

# Write CSV
np.savetxt(
    Path(repo_root) / "derivatives" / "models" / "tdann_predictions.csv",
    pred_np,
    delimiter=",",
    fmt="%d"   # class indices are ints
)


t0 = time.time()
resnet_pred = []
for i,m in enumerate(resnet):
    t1 = time.time()
    print(f'{t1-t0}: Evaluationg mdl {i}')

    m.to(DEVICE).eval()
    torch.manual_seed(0)
    mdl_pred = []
    for j,(xb, _) in enumerate(dataloader):
        xb = xb.to(DEVICE, non_blocking=True)
        with torch.no_grad():
            this_pred = m(xb)[0].softmax(dim=1).argmax(dim=1)

        mdl_pred.append(this_pred.to('cpu'))
    resnet_pred.append(torch.cat(mdl_pred))

# Stack into (M, N) then transpose to (N, M)
resnet_pred_matrix = torch.stack(resnet_pred, dim=0).T   # shape: (N_stim, N_models)

# Move to CPU and NumPy
pred_np = resnet_pred_matrix.cpu().numpy()

# Write CSV
np.savetxt(
    Path(repo_root) / "derivatives" / "models" / "resnet_predictions.csv",
    pred_np,
    delimiter=",",
    fmt="%d"   # class indices are ints
)



## What follows below could have been a separate script. It's merged here for convenience

NLTK_DATA_DIR=Path(repo_root) / 'resources' / 'nltk_data'

nltk.download('wordnet', download_dir=NLTK_DATA_DIR)
nltk.data.path.insert(0,NLTK_DATA_DIR)

from nltk.corpus import wordnet as wn

def wnid_to_synset(wnid: str) -> Synset:
    wnid = wnid.strip()
    if "-" in wnid:
        offset_str, pos = wnid.split("-")
        return wn.synset_from_pos_and_offset(pos, int(offset_str))
    else:
        pos = wnid[0]
        offset_str = wnid[1:]
        return wn.synset_from_pos_and_offset(pos, int(offset_str))

@lru_cache(maxsize=None)
def _all_hypernym_ancestors_inclusive(s: Synset) -> frozenset[Synset]:
    """
    Return all hypernym ancestors of synset s, INCLUDING s itself.
    Uses transitive closure over hypernyms. Cached for speed.
    """
    anc = {s}
    # closure yields ancestors reachable by repeatedly applying hypernyms()
    for a in s.closure(lambda x: x.hypernyms()):
        anc.add(a)
    return frozenset(anc)


def count_imagenet_descendants(
    imagenet_wnids: Iterable[str],
    *,
    unique_leaves: bool = True,
    return_leaf_synsets: bool = False,
) -> Union[Dict[Synset, int], tuple[Dict[Synset, int], List[Synset]]]:
    """
    Count, for each WordNet node (Synset) a, how many ImageNet leaf classes
    descend from a, i.e. desc_count_IN(a).

    This *does not* attempt to count WordNet leaves; it counts only your provided
    ImageNet leaves.

    Args
    ----
    imagenet_wnids:
        Iterable of WNIDs (either 'n01440764' or '01440764-n').
    unique_leaves:
        If True, de-duplicate identical leaf synsets (recommended).
        If False, counts will reflect repeated wnids in input.
    return_leaf_synsets:
        If True, also returns the resolved leaf synsets list.

    Returns
    -------
    counts:
        dict mapping Synset -> number of ImageNet leaves under it
    (optional) leaf_synsets:
        list of Synset corresponding to the input wnids
    """
    leaf_synsets: List[Synset] = [wnid_to_synset(w) for w in imagenet_wnids]
    if unique_leaves:
        # preserve order but de-dup
        seen = set()
        deduped = []
        for s in leaf_synsets:
            if s not in seen:
                deduped.append(s)
                seen.add(s)
        leaf_synsets = deduped

    counts = Counter()
    for leaf in leaf_synsets:
        for a in _all_hypernym_ancestors_inclusive(leaf):
            counts[a] += 1

    counts_dict = dict(counts)
    if return_leaf_synsets:
        return counts_dict, leaf_synsets
    return counts_dict

def find_superclass(synset, counts, target_desc=10):
    candidates = []
    for path in synset.hypernym_paths():
        for a in path:
            n = counts.get(a, 0)  # number of ImageNet leaves under ancestor a
            if n >= target_desc:
                candidates.append((a, n))

    if not candidates:
        # fallback: if nothing meets target (rare), return the synset itself
        return synset

    # Choose the *lowest* ancestor that satisfies the criterion.
    # Original "min by n" picks the smallest group >= target_desc,
    # which is a reasonable proxy for "lowest" in the hierarchy.
    return min(candidates, key=lambda x: x[1])[0]


# reduce ImageNet labels to hypernym supersets

wnids_list = [wnid for k, wnid in wnids.items()]
counts = count_imagenet_descendants(wnids_list)  # dict: Synset -> int

leaf_to_super = {}
for w in wnids_list:
    leaf = wnid_to_synset(w)
    sup  = find_superclass(leaf, counts, target_desc=20)
    leaf_to_super[w] = sup  # store Synset, or sup.name() for a string key

supersets = set([s for _,s in leaf_to_super.items()])
n_superclasses = len(supersets)
print(f'Reduced {len(wnids_list)} imagenet synsets to {n_superclasses} supersets')

# get stats on superset membership

superset_counts = {s: [] for s in supersets}
superset_list = [s for _,s in leaf_to_super.items()]
for s in supersets:
    superset_counts[s]=sum([1 for this_s in superset_list if s == this_s])

print(superset_counts)

_,counts = zip(*list(superset_counts.items()))
print(f'Mean ImageNet synsets subsumed by supersets: {np.mean(counts)} +/- {np.std(counts)} (SD)')

# Evaluate ResNet label agreements 

from sklearn.metrics import f1_score

resnet_pred = np.loadtxt(Path(repo_root) / "derivatives" / "models" / "resnet_predictions.csv",
    delimiter=",")

n_pairs = int(resnet_pred.shape[1]/2)
f1 = []
resnet_acc, resnet_superset_acc = [], []
for i in range(n_pairs):
    j = i+n_pairs
    net1 = resnet_pred[:,i]
    net2 = resnet_pred[:,j]

    acc = 0
    for a, b in zip(net1, net2):
        if leaf_to_super[wnids[a]] == leaf_to_super[wnids[b]]:
            acc += 1

    f1.append(f1_score(net1, net2, average='macro'))
    resnet_acc.append(sum(net1 == net2))

    resnet_superset_acc.append(acc)

print('ResNet dyad aggreements (ImageNet labels):')
print([a/50000 for a in resnet_acc])
print('ResNet dyad aggreements (Hypernym superset labels):')
print([a/50000 for a in resnet_superset_acc])

# Evaluate TDANN label agreements

from sklearn.metrics import f1_score

tdann_pred = np.loadtxt(Path(repo_root) / "derivatives" / "models" / "tdann_predictions.csv",
    delimiter=",")

n_pairs = int(tdann_pred.shape[1]/2)
f1 = []
tdann_acc, tdann_superset_acc = [], []
for i in range(n_pairs):
    j = i+n_pairs
    net1 = tdann_pred[:,i]
    net2 = tdann_pred[:,j]

    acc = 0
    for a, b in zip(net1, net2):
        if leaf_to_super[wnids[a]] == leaf_to_super[wnids[b]]:
            acc += 1

    f1.append(f1_score(net1, net2, average='macro'))
    tdann_acc.append(sum(net1 == net2))

    tdann_superset_acc.append(acc)

print('TDANN dyad aggreements (ImageNet labels):')
print([a/50000 for a in tdann_acc])
print('TDANN dyad aggreements (Hypernym superset labels):')
print([a/50000 for a in tdann_superset_acc])