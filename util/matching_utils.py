"""Matching utils for computing point-to-point distances between two point clouds or meshes."""

import torch
import os
import sys
import torch
import yaml
import time
from tqdm import tqdm
from easydict import EasyDict as edict

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
import ot
from geomfum.refine import ZoomOut, NeuralZoomOut
from geomfum.convert import FmFromP2pConverter, NamFromP2pConverter, P2pFromNamConverter, GPUEuclideanNeighborFinder


from geomfum.descriptor.spectral import (
    WaveKernelSignature,
    LandmarkWaveKernelSignature,
)
from geomfum.descriptor.pipeline import (
    DescriptorPipeline,
    ArangeSubsampler,
    L2InnerNormalizer,
)


from scipy.optimize import linear_sum_assignment
from lapjv import lapjv
from torch.distributions import Normal

device = "cuda:0"
torch.cuda.set_device(device)


##############################   OURS AND VARIATIONS   #######################################


def compute_p2p_with_flows_composition(
    source_input,
    target_input,
    source_model,
    target_model,
    backward_steps: int,
    forward_steps: int,
):
    """
    Compute point-to-point maps using flow composition.

    Args:
        source_input: Source input tensor
        target_input: Target input tensor
        source_model: Source model
        target_model: Target model
        device: Device to run computations on
    """
    tqdm.write("> computing p2p with flows composition")
    start_time = time.time()

    with torch.no_grad():
        emb1_pullback = source_model.inverse(samples=source_input, num_steps=backward_steps)
        sample = target_model.sample(noise=emb1_pullback, num_steps=forward_steps)

    # Move to cpu for sklearn
    # sample_cpu = sample.cpu().numpy()
    # target_input_cpu = target_input.cpu().numpy()

    # nbrs = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(target_input_cpu)
    # _, p2p = nbrs.kneighbors(sample_cpu)
    # p2p = p2p[:, 0]

    dists = torch.cdist(sample, target_input)  # [N, M]
    p2p = torch.argmin(dists, dim=1).cpu().numpy()

    end_time = time.time()
    elapsed_time = end_time - start_time
    tqdm.write(f">>>> compute_p2p_with_flows_composition elapsed {elapsed_time:.4f}s")

    return p2p, elapsed_time

def compute_p2p_with_flows_composition_hungarian(
    source_input,
    target_input,
    source_model,
    target_model,
    backward_steps: int,
    forward_steps: int,
):
    """
    Compute point-to-point maps using flow composition with Hungarian algorithm.
    Args:
        source_input: Source input tensor
        target_input: Target input tensor
        source_model: Source model
        target_model: Target model
        backward_steps: Number of steps for inverse flow
        forward_steps: Number of steps for forward flow
    """
    tqdm.write("> computing p2p with flows composition + hungarian")
    start_time = time.time()
    with torch.no_grad():
        emb1_pullback = source_model.inverse(samples=source_input, num_steps=backward_steps)
        sample = target_model.sample(noise=emb1_pullback, num_steps=forward_steps)
    
    dists = torch.cdist(sample, target_input)
    row_ind, col_ind = linear_sum_assignment(dists.cpu().numpy())
    p2p = col_ind[np.argsort(row_ind)]
    end_time = time.time()
    elapsed_time = end_time - start_time
    tqdm.write(f">>>> compute_p2p_with_flows_composition_hungarian elapsed {elapsed_time:.4f}s")
    return p2p, elapsed_time


def compute_p2p_with_flows_composition_lapjv(
    source_input,
    target_input,
    source_model,
    target_model,
    backward_steps: int,
    forward_steps: int,
):
    """
    Compute point-to-point maps using flow composition with LAPJV algorithm.
    Args:
        source_input: Source input tensor
        target_input: Target input tensor
        source_model: Source model
        target_model: Target model
        backward_steps: Number of steps for inverse flow
        forward_steps: Number of steps for forward flow
    """
    tqdm.write("> computing p2p with flows composition + lapjv")
    start_time = time.time()
    with torch.no_grad():
        emb1_pullback = source_model.inverse(samples=source_input, num_steps=backward_steps)
        sample = target_model.sample(noise=emb1_pullback, num_steps=forward_steps)
    
    dists = torch.cdist(sample, target_input)
    row_ind, col_ind, _ = lapjv(dists.cpu().numpy())
    p2p = col_ind[np.argsort(row_ind)]
    end_time = time.time()
    elapsed_time = end_time - start_time
    tqdm.write(f">>>> compute_p2p_with_flows_composition_lapjv elapsed {elapsed_time:.4f}s")
    return p2p, elapsed_time


def compute_p2p_with_inverted_flows_in_gauss(
    source_input,
    target_input,
    source_model,
    target_model,
    backward_steps: int,
    forward_steps: int,
):
    """
    Compute point-to-point maps using nearest neighbor between the gaussians.
    Args:
        source_input: Source input tensor
        target_input: Target input tensor
        source_model: Source model
        target_model: Target model
    """
    tqdm.write("> computing p2p with inverted flows in gauss")
    start_time = time.time()
    with torch.no_grad():
        emb1_pullback = source_model.inverse(samples=source_input, num_steps=backward_steps)
        emb2_pullback = target_model.inverse(samples=target_input, num_steps=forward_steps)

    nbrs = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(
        emb2_pullback.cpu().numpy()
    )
    _, p2p = nbrs.kneighbors(emb1_pullback.cpu().numpy())
    p2p = p2p[:, 0]

    end_time = time.time()
    elapsed_time = end_time - start_time
    tqdm.write(f">>>> compute_p2p_with_inverted_flows_in_gauss elapsed {elapsed_time:.4f}s")

    return p2p, elapsed_time


def compute_p2p_with_inverted_flows_in_gauss_uniformed(
    source_input, target_input, source_model, target_model
):
    """
    Compute point-to-point maps using nearest neighbor between the gaussians.
    Args:
        source_input: Source input tensor
        target_input: Target input tensor
        source_model: Source model
        target_model: Target model
    """
    tqdm.write("> computing p2p with inverted flows in gauss uniformed")
    start_time = time.time()
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

    end_time = time.time()
    elapsed_time = end_time - start_time
    tqdm.write(
        f">>>> compute_p2p_with_inverted_flows_in_gauss_uniformed elapsed {elapsed_time:.4f}s"
    )

    return p2p, elapsed_time


def compute_p2p_with_flows_composition_zoomout(
    source_path,
    target_path,
    source_input,
    target_input,
    source_model,
    target_model,
    backward_steps: int,
    forward_steps: int,
    device: str,
):
    """
    Compute point-to-point maps using flow composition and zoomout

    Args:
        source_path: Path to the source mesh
        target_path: Path to the target mesh
        source_input: Source input tensor
        target_input: Target input tensor
        source_model: Source model
        target_model: Target model
        device: Device to run computations on
    """
    tqdm.write("> computing p2p with flows composition + zoomout")
    start_time = time.time()

    with torch.no_grad():
        emb1_pullback = source_model.inverse(samples=source_input, num_steps=backward_steps)
        sample = target_model.sample(noise=emb1_pullback, num_steps=forward_steps)

    # Move to cpu for sklearn
    sample_cpu = sample
    target_input_cpu = target_input

    p2p, _ = compute_p2p_with_knn_zoomout(
        source_path, target_path, sample_cpu, target_input_cpu,device
    )

    end_time = time.time()
    elapsed_time = end_time - start_time
    tqdm.write(f">>>> compute_p2p_with_flows_composition_zoomout elapsed {elapsed_time:.4f}s")

    return p2p, elapsed_time


def compute_p2p_with_flows_composition_neural_zoomout(
    source_path,
    target_path,
    source_input,
    target_input,
    source_model,
    target_model,
    backward_steps: int,
    forward_steps: int,
    device,
):
    """
    Compute point-to-point maps using flow composition and zoomout

    Args:
        source_path: Path to the source mesh
        target_path: Path to the target mesh
        source_input: Source input tensor
        target_input: Target input tensor
        source_model: Source model
        target_model: Target model
        device: Device to run computations on
    """
    tqdm.write("> computing p2p with flows composition + neural zoomout")
    start_time = time.time()

    with torch.no_grad():
        emb1_pullback = source_model.inverse(samples=source_input, num_steps=backward_steps)
        sample = target_model.sample(noise=emb1_pullback, num_steps=forward_steps)

    # Move to cpu for sklearn
    sample_cpu = sample
    target_input_cpu = target_input

    p2p, _ = compute_p2p_with_knn_neural_zoomout(
        source_path, target_path, sample_cpu, target_input_cpu
    )

    end_time = time.time()
    elapsed_time = end_time - start_time
    tqdm.write(
        f">>>> compute_p2p_with_flows_composition_neural_zoomout elapsed {elapsed_time:.4f}s"
    )

    return p2p, elapsed_time


################ KNN #######################


def compute_p2p_with_knn(source_input, target_input):
    """
    Compute point-to-point maps using nearest neighbor between features.
    Args:
        source_input: Source input tensor
        target_input: Target input tensor
    """
    tqdm.write("> computing p2p with knn")
    start_time = time.time()
    nbrs = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(
        target_input.cpu().numpy()
    )
    _, p2p = nbrs.kneighbors(source_input.cpu().numpy())
    p2p = p2p[:, 0]

    end_time = time.time()
    elapsed_time = end_time - start_time
    tqdm.write(f">>>> compute_p2p_with_knn elapsed {elapsed_time:.4f}s")

    return p2p, elapsed_time


################ LINEAR ASSIGNMENT #######################
def compute_p2p_with_hungarian(source_input, target_input):
    """
    Compute point-to-point maps using Hungarian algorithm between features.
    Args:
        source_input: Source input tensor
        target_input: Target input tensor
    """
    tqdm.write("> computing p2p with hungarian")
    start_time = time.time()
    # Compute the pairwise distances between source_input and target_input
    dists = torch.cdist(source_input, target_input)

    row_ind, col_ind = linear_sum_assignment(dists.cpu().numpy())
    p2p = col_ind[np.argsort(row_ind)]

    end_time = time.time()
    elapsed_time = end_time - start_time
    tqdm.write(f">>>> compute_p2p_with_hungarian elapsed {elapsed_time:.4f}s")

    return p2p, elapsed_time


def compute_p2p_with_lapjv(source_input, target_input):
    """
    Compute point-to-point maps using Hungarian algorithm between features.
    Args:
        source_input: Source input tensor
        target_input: Target input tensor
    """
    tqdm.write("> computing p2p with lapjv")
    start_time = time.time()
    # Compute the pairwise distances between source_input and target_input
    dists = torch.cdist(source_input, target_input)

    row_ind, col_ind, _ = lapjv(dists.cpu().numpy())
    p2p = col_ind[np.argsort(row_ind)]

    end_time = time.time()
    elapsed_time = end_time - start_time
    tqdm.write(f">>>> compute_p2p_with_lapjv elapsed {elapsed_time:.4f}s")

    return p2p, elapsed_time


################ FUNCTIONAL MAPS #######################


def compute_p2p_with_fmaps(source_path, target_path, source_features, target_features):
    """
    Compute point-to-point maps using functional maps optimized on initial features.
    Args:
        source_path: Path to the source mesh
        target_path: Path to the target mesh
        source_features: Source features (N, D)
        target_features: Target features (M, D)
    """
    tqdm.write("> computing p2p with fmaps")
    start_time = time.time()
    mesh = trimesh.load(source_path, process=False)
    mesh2 = trimesh.load(target_path, process=False)

    if len(mesh.faces) > 0:
        mesh_a = TriangleMesh(np.array(mesh.vertices), np.array(mesh.faces))
        mesh_b = TriangleMesh(np.array(mesh2.vertices), np.array(mesh2.faces))
    else:
        mesh_a = PointCloud(np.array(mesh.vertices))
        mesh_b = PointCloud(np.array(mesh2.vertices))

    mesh_a.laplacian.laplacian_finder = LaplacianFinder.from_registry(which="robust")
    mesh_b.laplacian.laplacian_finder = LaplacianFinder.from_registry(which="robust")

    mesh_a.laplacian.find_spectrum(spectrum_size=20)
    mesh_b.laplacian.find_spectrum(spectrum_size=20)
    mesh_b.basis.use_k = 20
    mesh_a.basis.use_k = 20

    descr_a = source_features.cpu().double().T
    descr_b = target_features.cpu().double().T

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

    end_time = time.time()
    elapsed_time = end_time - start_time
    tqdm.write(f">>>> compute_p2p_with_fmaps elapsed {elapsed_time:.4f}s")

    return p2p, elapsed_time


def compute_p2p_with_fmaps_wks(
    source_path, target_path, source_landmarks, target_landmarks
):
    """
    Compute point-to-point maps using functional maps.
    Args:
        source_path: Path to the source mesh
        target_path: Path to the target mesh
        input_landmark: Input landmark indices (n_ldmk)
        target_landmark: Target landmark indices (n_ldmk)
    """
    tqdm.write("> computing p2p with fmaps_wks")
    start_time = time.time()
    mesh = trimesh.load(source_path, process=False)
    mesh2 = trimesh.load(target_path, process=False)

    if len(mesh.faces) > 0:
        mesh_a = TriangleMesh(np.array(mesh.vertices), np.array(mesh.faces))
        mesh_b = TriangleMesh(np.array(mesh2.vertices), np.array(mesh2.faces))
    else:
        mesh_a = PointCloud(np.array(mesh.vertices))
        mesh_b = PointCloud(np.array(mesh2.vertices))

    mesh_a.laplacian.laplacian_finder = LaplacianFinder.from_registry(which="robust")
    mesh_b.laplacian.laplacian_finder = LaplacianFinder.from_registry(which="robust")

    mesh_a.laplacian.find_spectrum(spectrum_size=200)
    mesh_b.laplacian.find_spectrum(spectrum_size=200)
    mesh_b.basis.use_k = 200
    mesh_a.basis.use_k = 200

    mesh_a.landmark_indices = source_landmarks
    mesh_b.landmark_indices = target_landmarks

    steps = [
        WaveKernelSignature(n_domain=200, k=200),
        LandmarkWaveKernelSignature(n_domain=200, k=200),
        ArangeSubsampler(subsample_step=10),
        L2InnerNormalizer(),
    ]

    pipeline = DescriptorPipeline(steps)

    descr_a = pipeline.apply(mesh_a)
    descr_b = pipeline.apply(mesh_b)

    mesh_b.basis.use_k = 20
    mesh_a.basis.use_k = 20

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

    end_time = time.time()
    elapsed_time = end_time - start_time
    tqdm.write(f">>>> compute_p2p_with_fmaps_wks elapsed {elapsed_time:.4f}s")

    return p2p, elapsed_time


def compute_p2p_with_knn_zoomout(source_path, target_path, source_input, target_input, device="cuda:1"):
    """
    knn + zoomout
    Compute point-to-point maps using functional maps optimized on initial features.
    Args:
        source_path: Path to the source mesh
        target_path: Path to the target mesh
        source_input: Source features (N, D)
        target_input: Target features (M, D)
    """
    tqdm.write("> computing p2p with knn + zoomout")
    start_time = time.time()
    nbrs = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(
        target_input.cpu().numpy()
    )
    _, p2p = nbrs.kneighbors(source_input.cpu().numpy())
    p2p = p2p[:, 0]

    mesh = trimesh.load(source_path, process=False)
    mesh2 = trimesh.load(target_path, process=False)

    if len(mesh.faces) > 0:
        mesh_a = TriangleMesh(np.array(mesh.vertices), np.array(mesh.faces))
        mesh_b = TriangleMesh(np.array(mesh2.vertices), np.array(mesh2.faces))
    else:
        mesh_a = PointCloud(np.array(mesh.vertices))
        mesh_b = PointCloud(np.array(mesh2.vertices))

    mesh_a.laplacian.laplacian_finder = LaplacianFinder.from_registry()
    mesh_b.laplacian.laplacian_finder = LaplacianFinder.from_registry()

    mesh_a.laplacian.find_spectrum(spectrum_size=125)
    mesh_b.laplacian.find_spectrum(spectrum_size=125)

    mesh_a.basis.full_vecs = mesh_a.basis.full_vecs.to(device)
    mesh_b.basis.full_vecs = mesh_b.basis.full_vecs.to(device)
    mesh_a.basis.full_vals = mesh_a.basis.full_vals.to(device)
    mesh_b.basis.full_vals = mesh_b.basis.full_vals.to(device)
    
    mesh_a.laplacian._mass_matrix = mesh_a.laplacian._mass_matrix.to(device)
    mesh_b.laplacian._mass_matrix = mesh_b.laplacian._mass_matrix.to(device)
    
    mesh_b.basis.use_k = 20
    mesh_a.basis.use_k = 20

    p2p_to_fm = FmFromP2pConverter(pseudo_inverse=True)
    converter = P2pFromFmConverter(neighbor_finder=GPUEuclideanNeighborFinder())
    fmap = p2p_to_fm(p2p, mesh_b.basis, mesh_a.basis)

    
    mesh_b.basis.use_k = 125
    mesh_a.basis.use_k = 125
    zoomout = ZoomOut(nit=20, step=5, p2p_from_fm_converter=converter, fm_from_p2p_converter=p2p_to_fm)
    fmap = zoomout(fmap, mesh_b.basis, mesh_a.basis)

    p2p = converter(fmap, mesh_b.basis, mesh_a.basis)
    p2p = p2p.cpu().numpy()

    end_time = time.time()
    elapsed_time = end_time - start_time
    tqdm.write(f">>>> compute_p2p_with_knn_zoomout elapsed {elapsed_time:.4f}s")

    return p2p, elapsed_time


def compute_p2p_with_fmap_zoomout(source_path, target_path, source_input, target_input):
    """
    fmap + zoomout
    Compute point-to-point maps using functional maps optimized on initial features.
    Args:
        source_path: Path to the source mesh
        target_path: Path to the target mesh
        source_input: Source features (N, D)
        target_input: Target features (M, D)
    """
    tqdm.write("> computing p2p with fmap + zoomout")
    start_time = time.time()
    mesh = trimesh.load(source_path, process=False)
    mesh2 = trimesh.load(target_path, process=False)

    if len(mesh.faces) > 0:
        mesh_a = TriangleMesh(np.array(mesh.vertices), np.array(mesh.faces))
        mesh_b = TriangleMesh(np.array(mesh2.vertices), np.array(mesh2.faces))
    else:
        mesh_a = PointCloud(np.array(mesh.vertices))
        mesh_b = PointCloud(np.array(mesh2.vertices))

    mesh_a.laplacian.laplacian_finder = LaplacianFinder.from_registry(which="robust")
    mesh_b.laplacian.laplacian_finder = LaplacianFinder.from_registry(which="robust")

    mesh_a.laplacian.find_spectrum(spectrum_size=125)
    mesh_b.laplacian.find_spectrum(spectrum_size=125)
    mesh_b.basis.use_k = 20
    mesh_a.basis.use_k = 20

    descr_a = source_input.cpu().double().T
    descr_b = target_input.cpu().double().T

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
    
    fmap=fmap.to(device)

    mesh_a.basis.full_vecs = mesh_a.basis.full_vecs.to(device)
    mesh_b.basis.full_vecs = mesh_b.basis.full_vecs.to(device)
    mesh_a.basis.full_vals = mesh_a.basis.full_vals.to(device)
    mesh_b.basis.full_vals = mesh_b.basis.full_vals.to(device)
    
    mesh_a.laplacian._mass_matrix = mesh_a.laplacian._mass_matrix.to(device)
    mesh_b.laplacian._mass_matrix = mesh_b.laplacian._mass_matrix.to(device)
    mesh_b.basis.use_k = 20
    mesh_a.basis.use_k = 20

    p2p_to_fm = FmFromP2pConverter(pseudo_inverse=True)
    converter = P2pFromFmConverter(adjoint=False, neighbor_finder=GPUEuclideanNeighborFinder())
    mesh_b.basis.use_k = 125
    mesh_a.basis.use_k = 125

    zoomout = ZoomOut(nit=20, step=5, p2p_from_fm_converter=converter, fm_from_p2p_converter=p2p_to_fm)
    fmap = zoomout(fmap, mesh_b.basis, mesh_a.basis)
    p2p = converter(fmap, mesh_b.basis, mesh_a.basis)

    p2p = p2p.cpu().numpy()

    end_time = time.time()
    elapsed_time = end_time - start_time
    tqdm.write(f">>>> compute_p2p_with_fmap_zoomout elapsed {elapsed_time:.4f}s")

    return p2p, elapsed_time


def compute_p2p_with_knn_neural_zoomout(
    source_path, target_path, source_input, target_input, device="cuda:1"
):
    """
    knn + zoomout
    Compute point-to-point maps using functional maps optimized on initial features.
    Args:
        source_path: Path to the source mesh
        target_path: Path to the target mesh
        source_input: Source features (N, D)
        target_input: Target features (M, D)
    """
    tqdm.write("> computing p2p with knn + neural zoomout")
    start_time = time.time()
    nbrs = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(
        target_input.cpu().numpy()
    )
    _, p2p = nbrs.kneighbors(source_input.cpu().numpy())
    p2p = p2p[:, 0]

    mesh = trimesh.load(source_path, process=False)
    mesh2 = trimesh.load(target_path, process=False)

    if len(mesh.faces) > 0:
        mesh_a = TriangleMesh(np.array(mesh.vertices), np.array(mesh.faces))
        mesh_b = TriangleMesh(np.array(mesh2.vertices), np.array(mesh2.faces))
    else:
        mesh_a = PointCloud(np.array(mesh.vertices))
        mesh_b = PointCloud(np.array(mesh2.vertices))

    mesh_a.laplacian.laplacian_finder = LaplacianFinder.from_registry(which="robust")
    mesh_b.laplacian.laplacian_finder = LaplacianFinder.from_registry(which="robust")

    mesh_a.laplacian.find_spectrum(spectrum_size=125)
    mesh_b.laplacian.find_spectrum(spectrum_size=125)
    
    
    mesh_a.basis.full_vecs = mesh_a.basis.full_vecs.to(device)
    mesh_b.basis.full_vecs = mesh_b.basis.full_vecs.to(device)
    mesh_a.basis.full_vals = mesh_a.basis.full_vals.to(device)
    mesh_b.basis.full_vals = mesh_b.basis.full_vals.to(device)
    mesh_b.basis.use_k = 20
    mesh_a.basis.use_k = 20

    p2p_to_fm = NamFromP2pConverter(device=device)
    converter = P2pFromNamConverter(neighbor_finder=GPUEuclideanNeighborFinder())

    fmap = p2p_to_fm(p2p, mesh_b.basis, mesh_a.basis)

    mesh_a.basis.use_k = 125
    mesh_b.basis.use_k = 125
    zoomout = ZoomOut(nit=20, step=5, p2p_from_fm_converter=converter, fm_from_p2p_converter=p2p_to_fm)
    fmap = zoomout(fmap, mesh_b.basis, mesh_a.basis)
    p2p = converter(fmap, mesh_b.basis, mesh_a.basis)
    p2p = p2p.cpu().numpy()

    end_time = time.time()
    elapsed_time = end_time - start_time
    tqdm.write(f">>>> compute_p2p_with_knn_neural_zoomout elapsed {elapsed_time:.4f}s")

    return p2p, elapsed_time


def compute_p2p_with_fmap_neural_zoomout(
    source_path, target_path, source_input, target_input, device="cuda:1"
):
    """
    knn + neural_zoomout
    Compute point-to-point maps using functional maps optimized on initial features.
    Args:
        source_path: Path to the source mesh
        target_path: Path to the target mesh
        source_input: Source features (N, D)
        target_input: Target features (M, D)
    """
    tqdm.write("> computing p2p with fmap + neural zoomout")
    start_time = time.time()

    mesh = trimesh.load(source_path, process=False)
    mesh2 = trimesh.load(target_path, process=False)

    if len(mesh.faces) > 0:
        mesh_a = TriangleMesh(np.array(mesh.vertices), np.array(mesh.faces))
        mesh_b = TriangleMesh(np.array(mesh2.vertices), np.array(mesh2.faces))
    else:
        mesh_a = PointCloud(np.array(mesh.vertices))
        mesh_b = PointCloud(np.array(mesh2.vertices))

    mesh_a.laplacian.laplacian_finder = LaplacianFinder.from_registry(which="robust")
    mesh_b.laplacian.laplacian_finder = LaplacianFinder.from_registry(which="robust")

    mesh_a.laplacian.find_spectrum(spectrum_size=125)
    mesh_b.laplacian.find_spectrum(spectrum_size=125)
    

    mesh_b.basis.use_k = 20
    mesh_a.basis.use_k = 20

    descr_a = source_input.cpu().double().T
    descr_b = target_input.cpu().double().T

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

    p2p_from_fmap = P2pFromFmConverter(adjoint=True)
    p2p = p2p_from_fmap(fmap, mesh_b.basis, mesh_a.basis)
    
    
    nam_from_p2p = NamFromP2pConverter(device=device)
    fmap = nam_from_p2p(p2p, mesh_b.basis, mesh_a.basis)
    
    mesh_a.basis.full_vecs = mesh_a.basis.full_vecs.to(device)
    mesh_b.basis.full_vecs = mesh_b.basis.full_vecs.to(device)
    mesh_a.basis.full_vals = mesh_a.basis.full_vals.to(device)
    mesh_b.basis.full_vals = mesh_b.basis.full_vals.to(device)
    mesh_b.basis.use_k = 20
    mesh_a.basis.use_k = 20

    p2p_to_fm = NamFromP2pConverter(device=device)
    converter = P2pFromNamConverter(neighbor_finder=GPUEuclideanNeighborFinder())


    mesh_a.basis.use_k = 125
    mesh_b.basis.use_k = 125
    zoomout = ZoomOut(nit=20, step=5, p2p_from_fm_converter=converter, fm_from_p2p_converter=p2p_to_fm)
    fmap = zoomout(fmap, mesh_b.basis, mesh_a.basis)
    p2p = converter(fmap, mesh_b.basis, mesh_a.basis)

    p2p = p2p.cpu().numpy()
    end_time = time.time()
    elapsed_time = end_time - start_time
    tqdm.write(f">>>> compute_p2p_with_fmap_neural_zoomout elapsed {elapsed_time:.4f}s")

    return p2p, elapsed_time


################ Neural Deformation Pyramid #######################
# TODO: Make Shape_Matching_Baseline_wrapper a package to install via pip
_base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ndp_dir = os.path.normpath( os.path.join(_base_dir, "../", "Shape_Matching_Baseline_wrapper", "DeformationPyramid"))
sys.path.append(_ndp_dir)

from models.registration import Registration
from models.tiktok import Timers


def ndp_with_ldmks(source_path, target_path, source_landmarks, target_landmarks):
    """
    Compute point-to-point maps using neural deformation pyramids.
    Args:
        source_shape: Source shape
        target_shape: Target shape
        source_features: Source landmarks (source_n_landmarks)
        target_features: Target landmarks (target_n_landmarks)
    Returns:
        p2p: Point-to-point maps (source_n_points)
    """
    tqdm.write("> computing p2p with ndp_with_ldmks")
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

    with open(
        f"{_ndp_dir}/config/NDP.yaml",
        "r",
    ) as f:
        config = yaml.load(f, Loader=yaml.Loader)

    config = edict(config)

    # backup the experiment
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

    end_time = time.time()
    elapsed_time = end_time - start_time
    tqdm.write(f"> ndp_with_ldmks elapsed {elapsed_time:.4f}s")

    return p2p, elapsed_time


def compute_p2p_with_ndp_sdf(source_vertex, target_vertex, source_landmarks, target_landmarks):
    """
    Compute point-to-point maps using neural deformation pyramids.
    Args:
        source_sampled: Source sampled points (source_n_points, 3)
        target_sampled: Target sampled points (target_n_points, 3)
        source_vertex: Source shape vertices (source_n_vertices, 3)
        target_vertex: Target shape vertices (target_n_vertices, 3)
        source_landmarks: Source landmarks (source_n_landmarks)
        target_landmarks: Target landmarks (target_n_landmarks)
    Returns:
        p2p: Point-to-point maps (source_n_points)
    """
    tqdm.write("> computing p2p with ndp_with_ldmks")
    start_time = time.time()

    with open(
        f"{_ndp_dir}/config/NDP.yaml",
        "r",
    ) as f:
        config = yaml.load(f, Loader=yaml.Loader)

    config = edict(config)

    # backup the experiment
    if config.gpu_mode:
        config.device = torch.cuda.current_device()
    else:
        config.device = torch.device("cpu")

    model = Registration(config)
    timer = Timers()
    
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

    end_time = time.time()
    elapsed_time = end_time - start_time
    tqdm.write(f"> ndp_with_ldmks elapsed {elapsed_time:.4f}s")

    return p2p, elapsed_time


################ OT #######################
def compute_p2p_with_ot(source_features, target_features):
    """
    Compute point-to-point distances using sinkhorn optimal transport.

    Args:
        source_features: Source features (N, D)
        target_features: Target features (M, D)
    """
    tqdm.write("> computing p2p with ot (sinkhorn)")
    start_time = time.time()

    M = np.exp(-ot.dist(source_features.cpu().numpy(), target_features.cpu().numpy()))

    n, m = M.shape
    a = np.ones(n) / n
    b = np.ones(m) / m

    Gs = ot.sinkhorn(a, b, M, reg=1)

    indices = np.argsort(Gs, axis=1)[:, :1]
    p2p = indices.T[0, :]

    end_time = time.time()
    elapsed_time = end_time - start_time
    tqdm.write(f">>>> compute_p2p_with_ot elapsed {elapsed_time:.4f}s")

    return p2p, elapsed_time
