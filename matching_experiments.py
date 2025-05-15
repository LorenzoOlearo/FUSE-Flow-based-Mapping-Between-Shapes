import os
import numpy as np
import torch
import trimesh
from itertools import combinations
from sklearn.neighbors import NearestNeighbors
from util.matching_utils import (
    compute_p2p_with_geomdist,
    compute_p2p_with_knn_gauss,
    compute_p2p_with_knn,
    compute_p2p_with_fmaps
)
from util.mesh_utils import normalize_mesh
from model.models import FMCond
from model.networks import MLP_general

# Setup device
device = 'cuda:0'
torch.cuda.set_device(device)

# Constants
NUM_STEP = 64  # Number of steps for flow models
IN_DIR = '../datasets/meshes/SMAL_r/off/'
OUT_DIR = './smal_experiments/matching/'
CORR_DIR = '../datasets/meshes/SMAL_r/corres/'
CHECKPOINT_DIR = './smal_experiments/smal_landmarks_only/'
lm_smal=np.array([3162, 1931, 3731, 1399 ,1111, 1001])
lm=lm_smal
# Define landmarks (assuming lm_smal is defined elsewhere)
# You need to define lm_smal or load it from somewhere

# Get all file names in the directory
file_names = [f.split('.')[0] for f in os.listdir(IN_DIR) if f.endswith('.off')]

# Initialize result matrices
eucl = np.zeros((len(file_names), len(file_names)))
eucl_knn = np.zeros((len(file_names), len(file_names)))
eucl_knn_g = np.zeros((len(file_names), len(file_names)))
eucl_fm = np.zeros((len(file_names), len(file_names)))

# Iterate over combinations of file names
for source, target in combinations(file_names, 2):
    print(source, target)
    
    # Load meshes
    mesh1_path = IN_DIR+f'{source}.off'
    mesh2_path = IN_DIR+f'{target}.off'
    mesh1 = normalize_mesh(trimesh.load(mesh1_path, process=False))
    mesh2 = normalize_mesh(trimesh.load(mesh2_path, process=False))

    # Load correspondences
    corr_a = np.array(np.loadtxt(CORR_DIR+f'{source}.vts', dtype=np.int32)) - 1
    corr_b = np.array(np.loadtxt(CORR_DIR+f'{target}.vts', dtype=np.int32)) - 1
    
    # Target vertices for evaluation
    v2 = torch.tensor(mesh2.vertices).float().to(device)
    
    # Load features
    source_features = torch.tensor(np.loadtxt(CHECKPOINT_DIR+f'/{source}/features.txt').astype(np.float32)).to(device)
    target_features = torch.tensor(np.loadtxt(CHECKPOINT_DIR+f'/{target}/features.txt').astype(np.float32)).to(device)
    
    # Load models
    source_model = FMCond(channels=len(lm), network=MLP_general(channels=len(lm)).to(device))
    source_model.to(device)
    source_model.load_state_dict(torch.load(CHECKPOINT_DIR+f'/{source}/checkpoint-9999.pth', 
                                           map_location=device, weights_only=False)['model'], strict=True)
    
    target_model = FMCond(channels=len(lm), network=MLP_general(channels=len(lm)).to(device))
    target_model.to(device)
    target_model.load_state_dict(torch.load(CHECKPOINT_DIR+f'/{target}/checkpoint-9999.pth', 
                                           map_location=device, weights_only=False)['model'], strict=True)
    
    source_model.eval()
    target_model.eval()
    
    # 1. Compute with geometric distance
    p2p = compute_p2p_with_geomdist(source_features, target_features, source_model, target_model)
    eucl[file_names.index(source), file_names.index(target)] = torch.mean(torch.norm(v2.cpu()[p2p][corr_a] - v2.cpu()[corr_b], dim=-1)).item()
    print('Geomdist')
    print(eucl[file_names.index(source), file_names.index(target)])
    
    # 2. Compute with KNN
    p2p = compute_p2p_with_knn(source_features, target_features)
    eucl_knn[file_names.index(source), file_names.index(target)] = torch.mean(torch.norm(v2.cpu()[p2p][corr_a] - v2.cpu()[corr_b], dim=-1)).item()
    print('KNN')
    print(eucl_knn[file_names.index(source), file_names.index(target)])
    
    # 3. Compute with KNN in Gaussian space
    p2p = compute_p2p_with_knn_gauss(source_features, target_features, source_model, target_model)
    eucl_knn_g[file_names.index(source), file_names.index(target)] = torch.mean(torch.norm(v2.cpu()[p2p][corr_a] - v2.cpu()[corr_b], dim=-1)).item()
    print('KNN in Gauss')
    print(eucl_knn_g[file_names.index(source), file_names.index(target)])
    
    # 4. Compute with functional maps
    # Extract landmarks for functional maps
    lm_a = corr_a[lm_smal]
    lm_b = corr_b[lm_smal]
    
    p2p = compute_p2p_with_fmaps(mesh1_path, mesh2_path, lm_a, lm_b)
    eucl_fm[file_names.index(source), file_names.index(target)] = torch.mean(torch.norm(v2.cpu()[p2p][corr_a] - v2.cpu()[corr_b], dim=-1)).item()
    print('Functional maps')
    print(eucl_fm[file_names.index(source), file_names.index(target)])


    # Print mean results
print("Mean Geometric Distance:", np.mean(eucl[eucl > 0]))
print("Mean KNN Distance:", np.mean(eucl_knn[eucl_knn > 0]))
print("Mean KNN Gaussian Distance:", np.mean(eucl_knn_g[eucl_knn_g > 0]))
print("Mean Functional Maps Distance:", np.mean(eucl_fm[eucl_fm > 0]))
# Save the results if needed
os.makedirs(OUT_DIR, exist_ok=True)
np.save(OUT_DIR+'results_geomdist.npy', eucl)
np.save(OUT_DIR+'results_knn.npy', eucl_knn)
np.save(OUT_DIR+'results_knn_gauss.npy', eucl_knn_g)
np.save(OUT_DIR+'results_fmaps.npy', eucl_fm)