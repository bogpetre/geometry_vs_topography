# This script computes similarities of activations across model layers for a pair 
# of models. We assume the models are paired off by seed as 0/7, 1/8, 2/9, etc.
# which are the dyads we use for the paper. Which specific pair is evaluated
# is determined by the variable TASK_ID which can vary from 0-13. Adjust NUM_CPUs
# variable to correspond to your system. A GPU with ~10GB of memory is recommended,
# but shouldn't be required. Whether you evaluate TDANNs or vanilla ResNet-18's 
# depends on your TASK_ID variable. TASK_ID < 7 evaluates TDANN dyads. Otherwise
# You evaluate ResNet dyads.
#
# CKA is estimated using a minibatch approach which we repeat 10 times using 10
# precomputed shufflings of the dataset (for reproducability). Cosine similarity
# does not vary across shufflings, and is only computed once. Results are outputed 
# to derivatives/models/tdann_v_resnet_similarities_wide.csv
#
# This script will check if ImageNet data has been copied to scratch yet and if not
# will copy it there based on settings specified in config.json. To avoid file
# collisions, don't run these concurrently on the same compute node.
#
# This should take ~7 minutes to run on an A100 GPU with 32 CPUs/GPU

import os
import shutil
import json
import csv
from pathlib import Path
import warnings
import torch
import numpy as np

from numcodecs import Blosc

# next function is copied with modifications from the TDANN FeatureExtractor class in demo/src/features.py
def get_hook(layer, layer_name, target_dict, vectorize=False):
    def hook_function(_layer, _input, output, name=layer_name):
        out = output.detach()

        if vectorize:
            out = torch.reshape(out, (len(out), -1))
        target_dict[name] = out

    hook = layer.register_forward_hook(hook_function)
    return hook

def mean_cosine_similarity(X: torch.tensor, Y: torch.tensor, eps = 1e-12) -> torch.tensor:
    """
    Mean cosine similarity, computed per sample, between two representations.
    X, Y: (n_samples, n_features)
    """
    Xn = torch.nn.functional.normalize(X, p=2, dim=1, eps=eps)
    Yn = torch.nn.functional.normalize(Y, p=2, dim=1, eps=eps)

    cos = (Xn * Yn).sum(dim=1)  # (n_samples,)
    return cos.mean()

def HSIC_est(K,L):
    '''
    Unbiased HSIC estimator of Song et al. (2012) J Machine Learning Research
    K and L must have diagonal zero before running this
    '''
    assert K.shape == L.shape, 'HSIC_est(K,L) needs mini batch gram matrices of equal size'

    #assert not torch.any(torch.diag(K) != 0), 'K must have zeroed diagonals'
    #assert not torch.any(torch.diag(L) != 0), 'L must have zeroed diagonals'
    
    n1, m1 = K.shape
    assert n1 == m1, 'K must be square'
    #assert torch.allclose(K, K.T, atol=1e-6, rtol=0), 'K must be symmetric'

    n2, m2 = L.shape
    assert n2 == m2, 'L must be square'
    #assert torch.allclose(L, L.T, atol=1e-6, rtol=0), 'L must be symmetric'

    assert n1 == n2, 'K and L must have the same dimensions'
    assert n1 >= 4, 'Must have at least 4 observations for unbiased HSIC estimation'

    n = n1
    
    K1 = torch.sum(K, dim=1)
    L1 = torch.sum(L, dim=1)

    K_sum = torch.sum(K1)
    L_sum = torch.sum(L1)
    
    KL_sum = torch.dot(K1,L1)
    
    trKL = torch.sum(K*L) # == trace(K@L) if K is symmetric

    HSIC_1 = 1/(n*(n-3))*(trKL + K_sum*L_sum/((n-1)*(n-2)) - 2/(n-2)*KL_sum)

    return HSIC_1


NUM_CPUs = int(os.getenv('SLURM_CPUS_PER_TASK')) # replace this with a job scheduler environment variable
TASK_ID = int(os.getenv('SLURM_ARRAY_TASK_ID'))

file_path = Path(os.path.dirname(os.path.realpath(__file__)))
repo_root = file_path / '..'

config_path = repo_root / 'config.json'

with open(config_path) as f:
    config = json.load(f)

ST_BASE_FS = repo_root / 'resources' / 'ann_data'
os.environ['ST_BASE_FS'] = str(ST_BASE_FS)
#os.environ['CUDA_VISIBLE_DEVICES'] = "" # this is due to a different torch version that's not compatible with my local CUDA

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if DEVICE == 'cpu':
    warnings.warn('GPU not autodetected.')
else:
    print('Successfully found GPU')

TDANN_path = Path(config['TDANN_repo_path'])
os.chdir(str(TDANN_path)) # this is the main repo code, which we need for loading old model weights

import sys
sys.path.append(str(TDANN_path / 'demo')) # we reuse some code from here for loading old model weights
from src.model import load_model_from_checkpoint
from src.data import load_image, create_dataloader
from src.features import FeatureExtractor, resolve_sequential_module_from_str

if TASK_ID < 7:
    # Evaluate TDANNs
    TDANN = True
else:
    # Evaluate ResNet-18s without loss function
    TDANN = False

# Which networks do we compare?
total_net_dyad_count = 7
NET1 = TASK_ID % total_net_dyad_count
NET2 = NET1 + total_net_dyad_count
NETS = [NET1, NET2]

# These are obtained from https://image-net.org/
VAL_DATA = Path(config['imagenet_path']) / 'ILSVRC2012_img_val.tar'
META_DATA = Path(config['imagenet_path']) / 'ILSVRC2012_devkit_t12.tar.gz'
SCRATCH_DIR = Path(config['scratch_ssd']) if Path(config['scratch_ssd']) else Path(config['scratch'])
if not SCRATCH_DIR.exists():
    os.mkdir(SCRATCH_DIR)
if not (SCRATCH_DIR / 'ILSVRC2012_img_val.tar').exists():
    shutil.copy(VAL_DATA, SCRATCH_DIR / 'ILSVRC2012_img_val.tar')  
    VAL_DATA = SCRATCH_DIR / 'ILSVRC2012_img_val.tar'
if not (SCRATCH_DIR / 'ILSVRC2012_devkit_t12.tar.gz').exists():
    shutil.copy(META_DATA, SCRATCH_DIR / 'ILSVRC2012_devkit_t12.tar.gz')
    META_DATA = SCRATCH_DIR / 'ILSVRC2012_devkit_t12.tar.gz'

image_path = SCRATCH_DIR # modify this to copy the necessary files from wherever they're stored based on config.json

CKPT_ROOT = repo_root / 'derivatives' / 'models' / 'checkpoints'

def tdann_ckpt_path_for_seed(seed: int, spatial = True) -> Path:
    """Return path to model_final_checkpoint for a given TDANN seed (0–4)."""

    '''
    subdir = "simclr_spatial_resnet18_swappedon_SineGrating2019_isoswap_3_"

    if not spatial:
        subdir = subdir + "lw0_"

    if not spatial or seed != 0:
        subdir = subdir + f"seed_{seed}_"

    if spatial:
        subdir = subdir + "linear_eval_checkpoints"
        chkpt = CKPT_ROOT / "linear_eval" / subdir / "model_final_checkpoint_phase27.torch"
    else:
        subdir = subdir + "checkpoints"
        chkpt = CKPT_ROOT / subdir / "model_final_checkpoint_phase199.torch"
    '''

    subdir = f"spatial_resnet18_swappedon_SineGrating2019_isoswap_3_seed_{seed}"

    if spatial:
        subdir = 'simclr_' + subdir
        chkpt = CKPT_ROOT / subdir / "model_final_checkpoint_phase199.torch"
    else:
        subdir = 'simclr_non' + subdir
        chkpt = CKPT_ROOT / subdir / "model_final_checkpoint_phase199.torch"

    return chkpt

# Load the models
seeds = {'Spatial': list(range(14)), 'Nonspatial': list(range(14))}

models = []

for net in NETS:
    if TDANN:
        ckpt_path = tdann_ckpt_path_for_seed(net)
        print(f"Loading TDANN seed {net} from {ckpt_path}")
    else:
        ckpt_path = tdann_ckpt_path_for_seed(net, spatial=False)
        print(f"Loading ResNet-18 seed {net} from {ckpt_path}")
    
    model = load_model_from_checkpoint(str(ckpt_path))  # uses demo/src/model.py
    model.eval()
    models.append(model)

import torchvision.transforms as transforms
import torchvision.datasets as datasets

from torch.utils.data import TensorDataset, DataLoader, Subset

imgnet_train_mean = [0.485, 0.456, 0.406]
imgnet_train_std = [0.229, 0.224, 0.225]

imgnet_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=imgnet_train_mean,
        std=imgnet_train_std),
])

# loading the ImageNet test dataset:
dataset = datasets.ImageNet(root=os.path.join(image_path), split='val', transform=imgnet_transforms)

target_layers = [
    'layer1.0',
    'layer1.1',
    'layer2.0',
    'layer2.1',
    'layer3.0',
    'layer3.1',
    'layer4.0',
    'layer4.1',
]

# Compute random lists
'''
This code was run once and then frozen. We use the resultant random permutations of 50k indices for
reproducable sampling of the validation dataset. Dataloader does not behave consistently across
torchvision versions, which makes it problematic to use even with a seed.
'''

n_iter = 10
'''
rng = np.random.default_rng(seed=0)

rand_ind = []
for i in range(n_iter):
    rand_ind.append(rng.permutation(50000))
rand_ind = np.vstack(rand_ind)

# save random lists for reproducibility 
np.save(repo_root / 'resources' / 'ann_data' / 'shuffled_imagenet_eval_indices.npy', rand_ind)
'''

# load random lists
rand_ind = np.load(repo_root / 'resources' / 'ann_data' / 'shuffled_imagenet_eval_indices.npy')


# Compute similarity measures using minibatch CKA and mean cosine similarity over minibatches
CKA = {l: [] for l in target_layers}
cosim = {l: 0.0 for l in target_layers}

# add model hooks to retrieve activations
activations = []
hooks = []
for model in models:
    activations.append({l: None for l in target_layers})
    hooks.append([])
    for layer_name in target_layers:
        layer = resolve_sequential_module_from_str(model, layer_name)
        hook = get_hook(layer, layer_name, target_dict=activations[-1], vectorize=False)
        hooks[-1].append(hook)

try:
    assert(len(models) == 2)
    for model in models:
        model.to(DEVICE)
        model.eval()

    # use repeated minibatch CKA estimation over differing partitions of data to reduce estimator variance
    for subsample_ind, subsample in enumerate(rand_ind):
        print(f'Evaluated iter {len(CKA[target_layers[0]])}')
        subset = Subset(dataset, subsample.tolist())

        # we sample 500 at a time to divide 50000 stimuli without remainder so can easily average
        # across minibatch estimates witout weights for sample size imbalance on the last batch
        assert len(subsample) % 500 == 0
        subset_loader = DataLoader(subset, num_workers=NUM_CPUs, batch_size=500, shuffle=False, 
            pin_memory=True, persistent_workers=True)


        # minibatch accumulators
        HSIC_KL = {l: 0.0 for l in target_layers}
        HSIC_KK = {l: 0.0 for l in target_layers}
        HSIC_LL = {l: 0.0 for l in target_layers}

        num_batches = 0
        with torch.no_grad():
            # iterate over minibatches
            for xb, _ in subset_loader:
                xb = xb.to(DEVICE, non_blocking=True)
                
                for m_ind, model in enumerate(models):
                    # reset activations to clear memory of prior iterations
                    for layer_name in target_layers:
                        activations[m_ind][layer_name] = None
                    
                    model(xb)

                # compute HSIC estimates for each layer separately and accumulate them accordingly
                for layer_name in target_layers:
                    X = activations[0][layer_name]
                    Y = activations[1][layer_name]

                    assert X is not None and Y is not None, f"Missing activation for {layer_name}"

                    n_stim = xb.shape[0]

                    X = X.flatten(1).to(torch.float32)
                    Y = Y.flatten(1).to(torch.float32)

                    K = torch.matmul(X,X.T)
                    L = torch.matmul(Y,Y.T)

                    K.fill_diagonal_(0)
                    L.fill_diagonal_(0)

                    HSIC_KL[layer_name] += HSIC_est(K,L)
                    HSIC_KK[layer_name] += HSIC_est(K,K)
                    HSIC_LL[layer_name] += HSIC_est(L,L)

                    # for one data partition, compute cosine similarity measures.
                    # Unlike CKA, these don't depend on data partitioning
                    if subsample_ind == 0:
                        cosim[layer_name] += mean_cosine_similarity(X, Y).detach().cpu().numpy()
                
                num_batches += 1

        # compute CKA estimate from HSICs and cosine similarity for each layer
        for layer_name in target_layers:
            _KL = HSIC_KL[layer_name] / num_batches
            _KK = HSIC_KK[layer_name] / num_batches
            _LL = HSIC_LL[layer_name] / num_batches

            denom = torch.sqrt(_KK)*torch.sqrt(_LL) + 1e-12
            CKA[layer_name].append((_KL / denom).to('cpu'))

            if subsample_ind == 0:
                cosim[layer_name] = cosim[layer_name] / num_batches
finally:
    for i in range(len(models)):
        for hook in hooks[i]:
            hook.remove()

# average CKA estimates over iterations for variance reduction
CKA_est = {}
for layer_name in target_layers:
    CKA_est[layer_name] = np.mean(torch.stack(CKA[layer_name]).cpu().numpy())


# write out data
out_file = repo_root / 'derivatives' / 'models' / 'tdann_v_resnet_similarities_wide.csv'

if TDANN:
    network_type = 'TDANN'
else:
    network_type = 'ResNet'

header = ['type','metric','seed1','seed2'] + list(target_layers)
cka_row = [network_type, 'CKA', NET1, NET2] + [CKA_est[layer_name] for layer_name in target_layers]
cosim_row = [network_type, 'cosim', NET1, NET2] + [cosim[layer_name] for layer_name in target_layers]

write_header = (not os.path.exists(out_file)) or (os.path.getsize(out_file) == 0)
with open(out_file, "a", newline="") as f:
    w = csv.writer(f)
    if write_header:
        w.writerow(header)
    w.writerow(cosim_row)
    w.writerow(cka_row)

print(f'Wrote similarity metrics to {out_file}')