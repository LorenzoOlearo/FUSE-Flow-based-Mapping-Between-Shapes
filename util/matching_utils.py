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
from util.mesh_utils import normalize_mesh_08
import ot
from geomfum.refine import ZoomOut, NeuralZoomOut
from geomfum.convert import FmFromP2pConverter, NamFromP2pConverter, P2pFromNamConverter


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
    source_input, target_input, source_model, target_model, device="cuda:1"
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


def compute_p2p_with_flows_composition_hungarian(
    source_input, target_input, source_model, target_model
):
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


def compute_p2p_with_flows_composition_lapjv(
    source_input, target_input, source_model, target_model
):
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


def compute_p2p_with_flows_composition_hungarian_optimized(
    source_input, target_input, source_model, target_model
):
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


def compute_p2p_with_inverted_flows_in_gauss(
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
    with torch.no_grad():
        emb1_pullback = source_model.inverse(samples=source_input, num_steps=64)
        emb2_pullback = target_model.inverse(samples=target_input, num_steps=64)

    nbrs = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(
        emb2_pullback.cpu().numpy()
    )
    _, p2p = nbrs.kneighbors(emb1_pullback.cpu().numpy())
    p2p = p2p[:, 0]

    return p2p


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


def compute_p2p_with_flows_composition_zoomout(
    source_path,
    target_path,
    source_input,
    target_input,
    source_model,
    target_model,
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
    source_input = source_input.to(device)
    target_input = target_input.to(device)
    source_model = source_model.to(device)
    target_model = target_model.to(device)

    with torch.no_grad():
        emb1_pullback = source_model.inverse(samples=source_input, num_steps=64)
        sample = target_model.sample(noise=emb1_pullback, num_steps=64)

    # Move to cpu for sklearn
    sample_cpu = sample
    target_input_cpu = target_input

    p2p = compute_p2p_with_knn_zoomout(
        source_path, target_path, sample_cpu, target_input_cpu
    )

    return p2p


def compute_p2p_with_flows_composition_neural_zoomout(
    source_path,
    target_path,
    source_input,
    target_input,
    source_model,
    target_model,
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
    source_input = source_input.to(device)
    target_input = target_input.to(device)
    source_model = source_model.to(device)
    target_model = target_model.to(device)

    with torch.no_grad():
        emb1_pullback = source_model.inverse(samples=source_input, num_steps=64)
        sample = target_model.sample(noise=emb1_pullback, num_steps=64)

    # Move to cpu for sklearn
    sample_cpu = sample
    target_input_cpu = target_input

    p2p = compute_p2p_with_knn_neural_zoomout(
        source_path, target_path, sample_cpu, target_input_cpu
    )

    return p2p


################ KNN #######################


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


################ LINEAR ASSIGNMENT #######################
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
    mesh = trimesh.load(source_path, process=False)
    mesh2 = trimesh.load(target_path, process=False)

    if len(mesh.faces) > 0:
        mesh = normalize_mesh_08(trimesh.load(source_path, process=False))
        mesh2 = normalize_mesh_08(trimesh.load(target_path, process=False))

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


def compute_p2p_with_fmaps_wks(source_path, target_path, source_ldm, target_landmark):
    """
    Compute point-to-point maps using functional maps.
    Args:
        source_path: Path to the source mesh
        target_path: Path to the target mesh
        input_landmark: Input landmark indices (n_ldmk)
        target_landmark: Target landmark indices (n_ldmk)
    """
    mesh = trimesh.load(source_path, process=False)
    mesh2 = trimesh.load(target_path, process=False)

    if len(mesh.faces) > 0:
        mesh = normalize_mesh_08(trimesh.load(source_path, process=False))
        mesh2 = normalize_mesh_08(trimesh.load(target_path, process=False))

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

    mesh_a.landmark_indices = source_ldm
    mesh_b.landmark_indices = target_landmark

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

    p2p = converter(fmap, mesh_a.basis, mesh_b.basis)

    return p2p


def compute_p2p_with_knn_zoomout(source_path, target_path, source_input, target_input):
    """
    knn + zoomout
    Compute point-to-point maps using functional maps optimized on initial features.
    Args:
        source_path: Path to the source mesh
        target_path: Path to the target mesh
        source_input: Source features (N, D)
        target_input: Target features (M, D)
    """
    nbrs = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(
        target_input.cpu().numpy()
    )
    _, p2p = nbrs.kneighbors(source_input.cpu().numpy())
    p2p = p2p[:, 0]

    mesh = trimesh.load(source_path, process=False)
    mesh2 = trimesh.load(target_path, process=False)

    if len(mesh.faces) > 0:
        mesh = normalize_mesh_08(trimesh.load(source_path, process=False))
        mesh2 = normalize_mesh_08(trimesh.load(target_path, process=False))

        mesh_a = TriangleMesh(np.array(mesh.vertices), np.array(mesh.faces))
        mesh_b = TriangleMesh(np.array(mesh2.vertices), np.array(mesh2.faces))
    else:
        mesh_a = PointCloud(np.array(mesh.vertices))
        mesh_b = PointCloud(np.array(mesh2.vertices))

    mesh_a.laplacian.laplacian_finder = LaplacianFinder.from_registry(which="robust")
    mesh_b.laplacian.laplacian_finder = LaplacianFinder.from_registry(which="robust")

    mesh_a.laplacian.find_spectrum(spectrum_size=200)
    mesh_b.laplacian.find_spectrum(spectrum_size=200)
    mesh_b.basis.use_k = 20
    mesh_a.basis.use_k = 20

    p2p_to_fm = FmFromP2pConverter()
    fmap_ini = p2p_to_fm(p2p, mesh_b.basis, mesh_a.basis)

    zoomout = ZoomOut(nit=20, step=5)
    fmap = zoomout(fmap_ini, mesh_b.basis, mesh_a.basis)
    converter = P2pFromFmConverter()
    p2p = converter(fmap, mesh_b.basis, mesh_a.basis)

    return p2p


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
    mesh = trimesh.load(source_path, process=False)
    mesh2 = trimesh.load(target_path, process=False)

    if len(mesh.faces) > 0:
        mesh = normalize_mesh_08(trimesh.load(source_path, process=False))
        mesh2 = normalize_mesh_08(trimesh.load(target_path, process=False))

        mesh_a = TriangleMesh(np.array(mesh.vertices), np.array(mesh.faces))
        mesh_b = TriangleMesh(np.array(mesh2.vertices), np.array(mesh2.faces))
    else:
        mesh_a = PointCloud(np.array(mesh.vertices))
        mesh_b = PointCloud(np.array(mesh2.vertices))

    mesh_a.laplacian.laplacian_finder = LaplacianFinder.from_registry(which="robust")
    mesh_b.laplacian.laplacian_finder = LaplacianFinder.from_registry(which="robust")

    mesh_a.laplacian.find_spectrum(spectrum_size=200)
    mesh_b.laplacian.find_spectrum(spectrum_size=200)
    mesh_b.basis.use_k = 20
    mesh_a.basis.use_k = 20

    descr_a = source_input.cpu().numpy().astype(np.float32).T
    descr_b = target_input.cpu().numpy().astype(np.float32).T

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
    mesh_a.basis.use_k = 200
    mesh_b.basis.use_k = 200
    zoomout = ZoomOut(nit=20, step=5)
    fmap = zoomout(fmap, mesh_b.basis, mesh_a.basis)
    converter = P2pFromFmConverter()
    p2p = converter(fmap, mesh_b.basis, mesh_a.basis)

    return p2p


def compute_p2p_with_knn_neural_zoomout(
    source_path, target_path, source_input, target_input
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
    nbrs = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(
        target_input.cpu().numpy()
    )
    _, p2p = nbrs.kneighbors(source_input.cpu().numpy())
    p2p = p2p[:, 0]

    mesh = trimesh.load(source_path, process=False)
    mesh2 = trimesh.load(target_path, process=False)

    if len(mesh.faces) > 0:
        mesh = normalize_mesh_08(trimesh.load(source_path, process=False))
        mesh2 = normalize_mesh_08(trimesh.load(target_path, process=False))

        mesh_a = TriangleMesh(np.array(mesh.vertices), np.array(mesh.faces))
        mesh_b = TriangleMesh(np.array(mesh2.vertices), np.array(mesh2.faces))
    else:
        mesh_a = PointCloud(np.array(mesh.vertices))
        mesh_b = PointCloud(np.array(mesh2.vertices))

    mesh_a.laplacian.laplacian_finder = LaplacianFinder.from_registry(which="robust")
    mesh_b.laplacian.laplacian_finder = LaplacianFinder.from_registry(which="robust")

    mesh_a.laplacian.find_spectrum(spectrum_size=200)
    mesh_b.laplacian.find_spectrum(spectrum_size=200)
    mesh_b.basis.use_k = 20
    mesh_a.basis.use_k = 20

    p2p_to_fm = NamFromP2pConverter()
    fmap_ini = p2p_to_fm(p2p, mesh_b.basis, mesh_a.basis)

    zoomout = NeuralZoomOut(nit=20, step=5, device="cuda:0")
    fmap = zoomout(fmap_ini, mesh_b.basis, mesh_a.basis)
    converter = P2pFromNamConverter()
    p2p = converter(fmap, mesh_b.basis, mesh_a.basis)

    return p2p


def compute_p2p_with_fmap_neural_zoomout(
    source_path, target_path, source_input, target_input
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
    source_input = source_input.cpu().numpy()
    target_input = target_input.cpu().numpy()
    
    mesh = trimesh.load(source_path, process=False)
    mesh2 = trimesh.load(target_path, process=False)

    if len(mesh.faces) > 0:
        mesh = normalize_mesh_08(trimesh.load(source_path, process=False))
        mesh2 = normalize_mesh_08(trimesh.load(target_path, process=False))

        mesh_a = TriangleMesh(np.array(mesh.vertices), np.array(mesh.faces))
        mesh_b = TriangleMesh(np.array(mesh2.vertices), np.array(mesh2.faces))
    else:
        mesh_a = PointCloud(np.array(mesh.vertices))
        mesh_b = PointCloud(np.array(mesh2.vertices))

    mesh_a.laplacian.laplacian_finder = LaplacianFinder.from_registry(which="robust")
    mesh_b.laplacian.laplacian_finder = LaplacianFinder.from_registry(which="robust")

    mesh_a.laplacian.find_spectrum(spectrum_size=200)
    mesh_b.laplacian.find_spectrum(spectrum_size=200)
    mesh_b.basis.use_k = 20
    mesh_a.basis.use_k = 20

    descr_a = source_input.T
    descr_b = target_input.T

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
    p2p_from_fmap = P2pFromFmConverter()
    p2p = p2p_from_fmap(fmap, mesh_b.basis, mesh_a.basis)
    nam_from_p2p = NamFromP2pConverter(device="cuda:0")
    fmap = nam_from_p2p(p2p, mesh_b.basis, mesh_a.basis)
    mesh_a.basis.use_k = 200
    mesh_b.basis.use_k = 200
    zoomout = NeuralZoomOut(nit=20, step=5, device="cuda:0")
    fmap = zoomout(fmap, mesh_b.basis, mesh_a.basis)
    converter = P2pFromNamConverter()
    p2p = converter(fmap, mesh_b.basis, mesh_a.basis)

    return p2p


################ Neural Deformation Pyramid #######################
# import sys

# sys.path.append(
#     "/home/ubuntu/giulio_vigano/SM-baselines/Shape_Matching_Baseline_wrapper/DeformationPyramid/"
# )
# import torch
# from models.registration import Registration
# import yaml
# from easydict import EasyDict as edict
# from models.tiktok import Timers


def ndp_with_ldmks(source_shape, target_shape, source_landmarks, target_landmarks):
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
    
    source_shape = TriangleMesh(
        np.array(source_shape.vertices), np.array(source_shape.faces)
    )
    target_shape = TriangleMesh(
        np.array(target_shape.vertices), np.array(target_shape.faces)
    )

    source_shape.landmark_indices = source_landmarks
    target_shape.landmark_indices = target_landmarks

    with open(
        "/home/ubuntu/giulio_vigano/SM-baselines/Shape_Matching_Baseline_wrapper/DeformationPyramid/config/NDP.yaml",
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
        torch.tensor(source_shape.vertices).float().to("cuda"),
        torch.tensor(target_shape.vertices).float().to("cuda"),
    )

    model.load_pcds(
        x, y, [x[source_shape.landmark_indices], y[target_shape.landmark_indices]]
    )

    timer.tic("registration")
    warped_pcd, iter_cnt, timer = model.register(timer=timer)
    timer.toc("registration")

    nbrs = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(y.cpu().numpy())
    _, p2p = nbrs.kneighbors(warped_pcd.cpu().numpy())
    p2p = p2p[:, 0]

    return p2p


################ OT #######################


def compute_p2p_with_ot(source_features, target_features):
    """
    Compute point-to-point distances using sinkhorn optimal transport.

    Args:
        source_features: Source features (N, D)
        target_features: Target features (M, D)
    """

    M = np.exp(-ot.dist(source_features.cpu().numpy(), target_features.cpu().numpy()))

    n, m = M.shape
    a = np.ones(n) / n
    b = np.ones(m) / m

    Gs = ot.sinkhorn(a, b, M, reg=1)

    indices = np.argsort(Gs, axis=1)[:, :1]

    return indices.T[0, :]
