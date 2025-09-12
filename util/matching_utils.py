"""Matching utils for computing point-to-point distances between two point clouds or meshes."""

import torch
import sys

sys.path.append("..")
import numpy as np
import trimesh
from sklearn.neighbors import NearestNeighbors

from geomfum.functional_map import (
    FactorSum,
    LBCommutativityEnforcing,
    OperatorCommutativityEnforcing,
    SpectralDescriptorPreservation,
)
from geomfum.numerics.optimization import ScipyMinimize
from geomfum.shape import TriangleMesh, PointCloud
from geomfum.convert import P2pFromFmConverter
from geomfum.laplacian import LaplacianFinder
from util.mesh_utils import normalize_mesh_08 as normalize_mesh
import ot

from torch_linear_assignment import batch_linear_assignment
from torch_linear_assignment import assignment_to_indices

from scipy.optimize import linear_sum_assignment
from lapjv import lapjv
from torch.distributions import Normal

device = "cuda:0"
torch.cuda.set_device(device)


def compute_p2p_with_flows_composition(source_input, target_input, source_model, target_model, device):
    """
    Compute point-to-point maps using flow composition.

    Args:
        source_input: Source input tensor
        target_input: Target input tensor
        source_model: Source model
        target_model: Target model
        device: Device to run computations on
    """
    source_input = source_input.to(device)
    target_input = target_input.to(device)
    source_model = source_model.to(device)
    target_model = target_model.to(device)

    with torch.no_grad():
        emb1_pullback = source_model.inverse(samples=source_input, num_steps=64)
        sample = target_model.sample(noise=emb1_pullback, num_steps=64)

    # Move to cpu for sklearn
    sample_cpu = sample.cpu().numpy()
    target_input_cpu = target_input.cpu().numpy()

    nbrs = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(target_input_cpu)
    _, p2p = nbrs.kneighbors(sample_cpu)
    p2p = p2p[:, 0]

    return p2p


def compute_p2p_with_flows_composition_hungarian(source_input, target_input, source_model, target_model):
    """
    Compute point-to-point maps using flow composition with Hungarian algorithm.

    Args:
        source_input: Source input tensor
        target_input: Target input tensor
        source_model: Source model
        target_model: Target model
    """
    with torch.no_grad():
        emb1_pullback = source_model.inverse(samples=source_input, num_steps=64)
        sample = target_model.sample(noise=emb1_pullback, num_steps=64)

    # Compute the pairwise distances between sample and target_input
    dists = torch.cdist(sample, target_input)

    # Optimal assignment using Hungarian algorithm
    # assignment = batch_linear_assignment(dists)
    # row_ind, col_ind = assignment_to_indices(assignment)

    row_ind, col_ind = linear_sum_assignment(dists.cpu().numpy())

    p2p = col_ind[np.argsort(row_ind)]

    return p2p


def compute_p2p_with_flows_composition_lapjv(source_input, target_input, source_model, target_model):
    """
    Compute point-to-point maps using flow composition with Hungarian algorithm.

    Args:
        source_input: Source input tensor
        target_input: Target input tensor
        source_model: Source model
        target_model: Target model
    """
    with torch.no_grad():
        emb1_pullback = source_model.inverse(samples=source_input, num_steps=64)
        sample = target_model.sample(noise=emb1_pullback, num_steps=64)

    # Compute the pairwise distances between sample and target_input
    dists = torch.cdist(sample, target_input)

    row_ind, col_ind, _ = lapjv(dists.cpu().numpy())
    p2p = col_ind[np.argsort(row_ind)]

    return p2p


def compute_p2p_with_flows_composition_hungarian_optimized(source_input, target_input, source_model, target_model):
    
    with torch.no_grad():
        emb1_pullback = source_model.inverse(samples=source_input, num_steps=64)
        sample = target_model.sample(noise=emb1_pullback, num_steps=64)

    dists = torch.cdist(sample, target_input)
    nearest = torch.argmin(dists, dim=1)

    assigned = set()
    p2p = torch.empty(len(sample), dtype=torch.long)

    for i, j in sorted(enumerate(nearest), key=lambda x: dists[x[0], x[1]].item()):
        if j.item() not in assigned:
            p2p[i] = j
            assigned.add(j.item())
        else:
            # pick next best target not yet assigned
            candidates = torch.argsort(dists[i])
            for cand in candidates:
                if cand.item() not in assigned:
                    p2p[i] = cand
                    assigned.add(cand.item())
                    break
    return p2p


def compute_p2p_with_inverted_flows_in_gauss(source_input, target_input, source_model, target_model):
    """
    Compute point-to-point maps using nearest neighbor between the gaussians.
    Args:
        source_input: Source input tensor
        target_input: Target input tensor
        source_model: Source model
        target_model: Target model
    """
    with torch.no_grad():
        emb1_pullback = source_model.inverse(samples=source_input, num_steps=64)
        emb2_pullback = target_model.inverse(samples=target_input, num_steps=64)

    nbrs = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(
        emb2_pullback.cpu().numpy()
    )
    _, p2p = nbrs.kneighbors(emb1_pullback.cpu().numpy())
    p2p = p2p[:, 0]

    return p2p


def compute_p2p_with_inverted_flows_in_gauss_uniformed(source_input, target_input, source_model, target_model):
    """
    Compute point-to-point maps using nearest neighbor between the gaussians.
    Args:
        source_input: Source input tensor
        target_input: Target input tensor
        source_model: Source model
        target_model: Target model
    """
    with torch.no_grad():
        emb1_pullback = source_model.inverse(samples=source_input, num_steps=64)
        emb2_pullback = target_model.inverse(samples=target_input, num_steps=64)

    normal = Normal(0, 1)
    emb1_pullback_uniform = normal.cdf(emb1_pullback)
    emb2_pullback_uniform = normal.cdf(emb2_pullback)

    nbrs = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(
        emb2_pullback_uniform.cpu().numpy()
    )
    _, p2p = nbrs.kneighbors(emb1_pullback_uniform.cpu().numpy())
    p2p = p2p[:, 0]

    return p2p


def compute_p2p_with_knn(source_input, target_input):
    """
    Compute point-to-point maps using nearest neighbor between features.
    Args:
        source_input: Source input tensor
        target_input: Target input tensor
    """
    nbrs = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(
        target_input.cpu().numpy()
    )
    _, p2p = nbrs.kneighbors(source_input.cpu().numpy())
    p2p = p2p[:, 0]

    return p2p


def compute_p2p_with_hungarian(source_input, target_input):
    """
    Compute point-to-point maps using Hungarian algorithm between features.
    Args:
        source_input: Source input tensor
        target_input: Target input tensor
    """
    # Compute the pairwise distances between source_input and target_input
    dists = torch.cdist(source_input, target_input)

    # Optimal assignment using Hungarian algorithm
    # assignment = batch_linear_assignment(dists)
    # row_ind, col_ind = assignment_to_indices(assignment)
    # p2p = col_ind[np.argsort(row_ind)]

    row_ind, col_ind = linear_sum_assignment(dists.cpu().numpy())
    p2p = col_ind[np.argsort(row_ind)]

    return p2p


def compute_p2p_with_lapjv(source_input, target_input):
    """
    Compute point-to-point maps using Hungarian algorithm between features.
    Args:
        source_input: Source input tensor
        target_input: Target input tensor
    """
    # Compute the pairwise distances between source_input and target_input
    dists = torch.cdist(source_input, target_input)

    # Optimal assignment using Hungarian algorithm
    # assignment = batch_linear_assignment(dists)
    # row_ind, col_ind = assignment_to_indices(assignment)
    # p2p = col_ind[np.argsort(row_ind)]

    row_ind, col_ind, _ = lapjv(dists.cpu().numpy())
    p2p = col_ind[np.argsort(row_ind)]

    return p2p


def compute_p2p_with_fmaps(source_path, target_path, source_features, target_features):
    """
    Compute point-to-point maps using functional maps.
    Args:
        source_path: Path to the source mesh
        target_path: Path to the target mesh
        source_features: Source features (N, D)
        target_features: Target features (M, D)
    """
    mesh = trimesh.load(source_path, process=False)
    mesh2 = trimesh.load(target_path, process=False)

    if len(mesh.faces) > 0:
        mesh = normalize_mesh(trimesh.load(source_path, process=False))
        mesh2 = normalize_mesh(trimesh.load(target_path, process=False))

        mesh_a = TriangleMesh(np.array(mesh.vertices), np.array(mesh.faces))
        mesh_b = TriangleMesh(np.array(mesh2.vertices), np.array(mesh2.faces))
    else:
        mesh_a = PointCloud(np.array(mesh.vertices))
        mesh_b = PointCloud(np.array(mesh2.vertices))

    mesh_a.laplacian.laplaceian_finder = LaplacianFinder.from_registry(which="robust")
    mesh_b.laplacian.laplaceian_finder = LaplacianFinder.from_registry(which="robust")

    mesh_a.laplacian.find_spectrum(spectrum_size=20)
    mesh_b.laplacian.find_spectrum(spectrum_size=20)
    mesh_b.basis.use_k = 20
    mesh_a.basis.use_k = 20

    descr_a = source_features.cpu().numpy().astype(np.float32).T
    descr_b = target_features.cpu().numpy().astype(np.float32).T

    factors = [
        SpectralDescriptorPreservation(
            mesh_b.basis.project(descr_b),
            mesh_a.basis.project(descr_a),
            weight=1.0,
        ),
        LBCommutativityEnforcing.from_bases(
            mesh_b.basis,
            mesh_a.basis,
            weight=1e-2,
        ),
        OperatorCommutativityEnforcing.from_multiplication(
            mesh_b.basis, descr_b, mesh_a.basis, descr_a, weight=1e-1
        ),
    ]

    objective = FactorSum(factors)

    x0 = np.zeros((mesh_a.basis.spectrum_size, mesh_b.basis.spectrum_size))
    optimizer = ScipyMinimize(
        method="L-BFGS-B",
    )

    res = optimizer.minimize(
        objective,
        x0,
        fun_jac=objective.gradient,
    )

    fmap = res.x.reshape(x0.shape)

    fmap.shape

    converter = P2pFromFmConverter()

    p2p = converter(fmap, mesh_b.basis, mesh_a.basis)

    return p2p


def compute_p2p_with_ot(source_features, target_features):
    """
    Compute point-to-point distances using sinkhorn optimal transport.

    Args:
        source_features: Source features (N, D)
        target_features: Target features (M, D)
    """

    M = np.exp(-ot.dist(source_features, target_features))

    n, m = M.shape
    a = np.ones(n) / n
    b = np.ones(m) / m

    Gs = ot.sinkhorn(a, b, M, reg=1)

    indices = np.argsort(Gs, axis=1)[:, :1]

    return indices.T[0, :]
