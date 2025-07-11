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
from model.networks import MLP

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import trimesh

# Setup device
device = 'cuda:1'
torch.cuda.set_device(device)

# Constants
NUM_STEP = 64  # Number of steps for flow models
IN_DIR = './data/test'
OUT_DIR = './smal_experiments/matching'
CORR_DIR = '../datasets/meshes/SMAL_r/corres/'
CHECKPOINT_DIR = './data/test'
# IN_DIR = '../datasets/meshes/SMAL_r/off/'
# OUT_DIR = './smal_experiments/matching/'
# CORR_DIR = '../datasets/meshes/SMAL_r/corres/'
# CHECKPOINT_DIR = './smal_experiments/smal_landmarks_only/'
# lm_smal=np.array([3162, 1931, 3731, 1399 ,1111, 1001])
# lm=lm_smal

lm = np.array([412, 5891,6593,3323,2119])

# Define landmarks (assuming lm_smal is defined elsewhere)
# You need to define lm_smal or load it from somewhere

# Get all file names in the directory
file_names = [f.split('.')[0] for f in os.listdir(IN_DIR) if f.endswith('.off')]

# Initialize result matrices
eucl = np.zeros((len(file_names), len(file_names)))
eucl_knn = np.zeros((len(file_names), len(file_names)))
eucl_knn_g = np.zeros((len(file_names), len(file_names)))
eucl_fm = np.zeros((len(file_names), len(file_names)))

source = 'tg_reg_002.off'
target = 'tg_reg_009.off'

# Iterate over combinations of file names
print(source, target)

# Load meshes
# mesh1_path = IN_DIR+f'{source}.off'
# mesh2_path = IN_DIR+f'{target}.off'
mesh_1_path = './data/test/tr_reg_002.off'
mesh_2_path = './data/test/tr_reg_009.off'
mesh1 = normalize_mesh(trimesh.load(mesh_1_path, process=False))
mesh2 = normalize_mesh(trimesh.load(mesh_2_path, process=False))

# Load correspondences
# corr_a = np.array(np.loadtxt(CORR_DIR+f'{source}.vts', dtype=np.int32)) - 1
# corr_b = np.array(np.loadtxt(CORR_DIR+f'{target}.vts', dtype=np.int32)) - 1

# Target vertices for evaluation
v2 = torch.tensor(mesh2.vertices).float().to(device)

# Load features
source_features = torch.tensor(np.loadtxt(f'data/test/tr_reg_002-sdf-dijkstra.txt').astype(np.float32)).to(device)
target_features = torch.tensor(np.loadtxt(f'data/test/tr_reg_009-sdf-dijkstra.txt').astype(np.float32)).to(device)

# Load models
source_model = FMCond(channels=len(lm), network=MLP(channels=len(lm)).to(device))
source_model.to(device)
# source_model.load_state_dict(torch.load(f'data/test/{source.split('.')[0]}/checkpoint-9999.pth')['model'], strict=True)
source_model.load_state_dict(torch.load('./data/test/tr_reg_002/checkpoint-9999.pth', weights_only=False)['model'], strict=True)

target_model = FMCond(channels=len(lm), network=MLP(channels=len(lm)).to(device))
target_model.to(device)
target_model.load_state_dict(torch.load('./data/test/tr_reg_009/checkpoint-9999.pth', weights_only=False)['model'], strict=True)

source_model.eval()
target_model.eval()

# 1. Compute with geometric distance
p2p = compute_p2p_with_geomdist(source_features, target_features, source_model, target_model)
eucl[0, 0] = torch.mean(torch.norm(v2.cpu()[p2p] - v2.cpu(), dim=-1)).item()
print(f"Ours: {eucl[0, 0]}")
# 2. Compute with KNN
p2p = compute_p2p_with_knn(source_features, target_features)
eucl_knn[0, 0] = torch.mean(torch.norm(v2.cpu()[p2p] - v2.cpu(), dim=-1)).item()
print(f"KNN: {eucl_knn[0, 0]}")
# 3. Compute with KNN in Gaussian space
p2p = compute_p2p_with_knn_gauss(source_features, target_features, source_model, target_model)
eucl_knn_g[0, 0] = torch.mean(torch.norm(v2.cpu()[p2p] - v2.cpu(), dim=-1)).item()
print(f"KNN in Gauss: {eucl_knn_g[0, 0]}")

# 4. Compute with functional maps
# Extract landmarks for functional maps
lm_a = lm
lm_b = lm

p2p = compute_p2p_with_fmaps(mesh_1_path, mesh_2_path, source_features, target_features)
eucl_fm[0, 0] = torch.mean(torch.norm(v2.cpu()[p2p] - v2.cpu(), dim=-1)).item()
print(f'Functional maps: {eucl_fm[0, 0]}')


# Print mean results
# print("Mean Geometric Distance:", np.mean(eucl[eucl > 0]))
# print("Mean KNN Distance:", np.mean(eucl_knn[eucl_knn > 0]))
# print("Mean KNN Gaussian Distance:", np.mean(eucl_knn_g[eucl_knn_g > 0]))
# print("Mean Functional Maps Distance:", np.mean(eucl_fm[eucl_fm > 0]))
# # Save the results if needed
# os.makedirs(OUT_DIR, exist_ok=True)
# np.save(OUT_DIR+'results_geomdist.npy', eucl)
# np.save(OUT_DIR+'results_knn.npy', eucl_knn)
# np.save(OUT_DIR+'results_knn_gauss.npy', eucl_knn_g)
# np.save(OUT_DIR+'results_fmaps.npy', eucl_fm)
