"""Neural Deformation Pyramid point-to-point matching methods."""

import os
import sys
import time

import numpy as np
import torch
import trimesh
import yaml
from easydict import EasyDict as edict
from geomfum.shape import PointCloud, TriangleMesh
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

# TODO: Make Shape_Matching_Baseline_wrapper a package to install via pip
_base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_ndp_dir = os.path.normpath(
    os.path.join(_base_dir, "../", "Shape_Matching_Baseline_wrapper", "DeformationPyramid")
)
sys.path.append(_ndp_dir)

from models.registration import Registration
from models.tiktok import Timers


def compute_p2p_ndp_landmarks(source_path, target_path, source_landmarks, target_landmarks):
    tqdm.write("> computing p2p with ndp landmarks")
    start_time = time.time()

    mesh = trimesh.load(source_path, process=False)
    mesh2 = trimesh.load(target_path, process=False)

    if len(mesh.faces) > 0:
        mesh_a = TriangleMesh(np.array(mesh.vertices), np.array(mesh.faces))
        mesh_b = TriangleMesh(np.array(mesh2.vertices), np.array(mesh2.faces))
    else:
        mesh_a = PointCloud(np.array(mesh.vertices))
        mesh_b = PointCloud(np.array(mesh2.vertices))

    mesh_a.landmark_indices = source_landmarks
    mesh_b.landmark_indices = target_landmarks

    with open(f"{_ndp_dir}/config/NDP.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config = edict(config)

    if config.gpu_mode:
        config.device = torch.cuda.current_device()
    else:
        config.device = torch.device("cpu")

    model = Registration(config)
    timer = Timers()
    x, y = (
        torch.tensor(mesh_a.vertices).double().to("cuda"),
        torch.tensor(mesh_b.vertices).double().to("cuda"),
    )

    model.load_pcds(x, y, [x[mesh_a.landmark_indices], y[mesh_b.landmark_indices]])

    timer.tic("registration")
    warped_pcd, iter_cnt, timer = model.register(timer=timer)
    timer.toc("registration")

    nbrs = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(y.cpu().numpy())
    _, p2p = nbrs.kneighbors(warped_pcd.cpu().numpy())
    p2p = p2p[:, 0]

    elapsed_time = time.time() - start_time
    tqdm.write(f">>>> compute_p2p_ndp_landmarks elapsed {elapsed_time:.4f}s")
    return p2p, elapsed_time


def compute_p2p_ndp_sdf(
    source_vertex, target_vertex, source_landmarks, target_landmarks
):
    tqdm.write("> computing p2p with ndp_with_ldmks")
    start_time = time.time()

    with open(f"{_ndp_dir}/config/NDP.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config = edict(config)

    if config.gpu_mode:
        config.device = torch.cuda.current_device()
    else:
        config.device = torch.device("cpu")

    model = Registration(config)
    timer = Timers()
    x, y = (
        torch.tensor(source_vertex).double().to("cuda"),
        torch.tensor(target_vertex).double().to("cuda"),
    )

    model.load_pcds(x, y, [x[source_landmarks], y[target_landmarks]])

    timer.tic("registration")
    warped_pcd, iter_cnt, timer = model.register(timer=timer)
    timer.toc("registration")

    nbrs = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(y.cpu().numpy())
    _, p2p = nbrs.kneighbors(warped_pcd.cpu().numpy())
    p2p = p2p[:, 0]

    elapsed_time = time.time() - start_time
    tqdm.write(f">>>> compute_p2p_ndp_sdf elapsed {elapsed_time:.4f}s")
    return p2p, elapsed_time
