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
import matplotlib.colors as mcolors

from util.plot import plot_points, start_end_subplot


def create_rgb_colormap(points):
    """
    Create a colormap for the points based on their coordinates.
    The color is determined by the distance from the origin.
    """
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    points = (points - points.min(axis=0)) / (points.max(axis=0) - points.min(axis=0))
    colors_hex = [mcolors.rgb2hex(c) for c in points]
    
    return colors_hex 


def main():
    device = 'cuda:1'
    torch.cuda.set_device(device)

    source = 'tg_reg_000.off'
    target = 'tg_reg_009.off'

    source = './data/MPI-FAUST/test/scans/test_scan_000.ply'
    target = './data/MPI-FAUST/test/scans/test_scan_009.ply'

    print(source, target)
    lm = np.array([412, 5891, 6593, 3323, 2119])

    # Load features
    source_features = torch.tensor(np.loadtxt(f'./out/SDFs/tr_reg_000/tr_reg_000-sdf-dijkstra-features.txt').astype(np.float32)).to(device)
    target_features = torch.tensor(np.loadtxt(f'./out/SDFs/tr_reg_009/tr_reg_009-sdf-dijkstra-features.txt').astype(np.float32)).to(device)

    # Load models
    source_model = FMCond(channels=len(lm), network=MLP(channels=len(lm)).to(device))
    target_model = FMCond(channels=len(lm), network=MLP(channels=len(lm)).to(device))
    source_model.load_state_dict(torch.load('./out/flows/tr_reg_000/checkpoint-9999.pth', weights_only=False)['model'], strict=True)
    target_model.load_state_dict(torch.load('./out/flows/tr_reg_009/checkpoint-9999.pth', weights_only=False)['model'], strict=True)
    source_model.to(device)
    target_model.to(device)
    source_model.eval()
    target_model.eval()

    source_points = torch.tensor(np.loadtxt(f'./out/SDFs/tr_reg_000/tr_reg_000-sdf-sampled-points.txt').astype(np.float32)).to(device)
    target_points = torch.tensor(np.loadtxt(f'./out/SDFs/tr_reg_009/tr_reg_009-sdf-sampled-points.txt').astype(np.float32)).to(device)
    
    p2p = compute_p2p_with_geomdist(source_features, target_features, source_model, target_model)
    # p2p = compute_p2p_with_knn(source_features, target_features)
    target_points = target_points[p2p]
    
    source_points = source_points[:100000]
    target_points = target_points[:100000]

    start_end_subplot(
        source_points, target_points,
        plots_path='./out',
        run_name=f'matching',
        show=False
    )

    print(f"saved to ./out/sdf-flow-{source}_{target}.html")



if __name__ == "__main__":
    main()
