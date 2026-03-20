"""Functional maps point-to-point matching methods."""

import time

import numpy as np
import trimesh
from geomfum.convert import (
    FmFromP2pConverter,
    GPUEuclideanNeighborFinder,
    NamFromP2pConverter,
    P2pFromFmConverter,
    P2pFromNamConverter,
)
from geomfum.descriptor.pipeline import (
    ArangeSubsampler,
    DescriptorPipeline,
    L2InnerNormalizer,
)
from geomfum.descriptor.spectral import LandmarkWaveKernelSignature, WaveKernelSignature
from geomfum.functional_map import (
    FactorSum,
    LBCommutativityEnforcing,
    OperatorCommutativityEnforcing,
    SpectralDescriptorPreservation,
)
from geomfum.laplacian import LaplacianFinder
from geomfum.numerics.optimization import ScipyMinimize
from geomfum.refine import ZoomOut
from geomfum.shape import PointCloud, TriangleMesh
from tqdm import tqdm


def compute_p2p_fmaps(source_path, target_path, source_features, target_features):
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
        LBCommutativityEnforcing.from_bases(mesh_b.basis, mesh_a.basis, weight=1e-2),
        OperatorCommutativityEnforcing.from_multiplication(
            mesh_b.basis, descr_b, mesh_a.basis, descr_a, weight=1e-1
        ),
    ]

    objective = FactorSum(factors)
    x0 = np.zeros((mesh_a.basis.spectrum_size, mesh_b.basis.spectrum_size))
    optimizer = ScipyMinimize(method="L-BFGS-B")
    res = optimizer.minimize(objective, x0, fun_jac=objective.gradient)

    fmap = res.x.reshape(x0.shape)
    converter = P2pFromFmConverter()
    p2p = converter(fmap, mesh_b.basis, mesh_a.basis)

    elapsed_time = time.time() - start_time
    tqdm.write(f">>>> compute_p2p_fmaps elapsed {elapsed_time:.4f}s")
    return p2p, elapsed_time


def compute_p2p_ndp_wks(
    source_path, target_path, source_landmarks, target_landmarks
):
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
        LBCommutativityEnforcing.from_bases(mesh_b.basis, mesh_a.basis, weight=1e-2),
        OperatorCommutativityEnforcing.from_multiplication(
            mesh_b.basis, descr_b, mesh_a.basis, descr_a, weight=1e-1
        ),
    ]

    objective = FactorSum(factors)
    x0 = np.zeros((mesh_a.basis.spectrum_size, mesh_b.basis.spectrum_size))
    optimizer = ScipyMinimize(method="L-BFGS-B")
    res = optimizer.minimize(objective, x0, fun_jac=objective.gradient)

    fmap = res.x.reshape(x0.shape)
    converter = P2pFromFmConverter()
    p2p = converter(fmap, mesh_b.basis, mesh_a.basis)

    elapsed_time = time.time() - start_time
    tqdm.write(f">>>> compute_p2p_ndp_wks elapsed {elapsed_time:.4f}s")
    return p2p, elapsed_time


def compute_p2p_fmaps_zoomout(source_path, target_path, source_input, target_input, device: str):
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
        LBCommutativityEnforcing.from_bases(mesh_b.basis, mesh_a.basis, weight=1e-2),
        OperatorCommutativityEnforcing.from_multiplication(
            mesh_b.basis, descr_b, mesh_a.basis, descr_a, weight=1e-1
        ),
    ]

    objective = FactorSum(factors)
    x0 = np.zeros((mesh_a.basis.spectrum_size, mesh_b.basis.spectrum_size))
    optimizer = ScipyMinimize(method="L-BFGS-B")
    res = optimizer.minimize(objective, x0, fun_jac=objective.gradient)

    fmap = res.x.reshape(x0.shape)
    fmap = fmap.to(device)

    mesh_a.basis.full_vecs = mesh_a.basis.full_vecs.to(device)
    mesh_b.basis.full_vecs = mesh_b.basis.full_vecs.to(device)
    mesh_a.basis.full_vals = mesh_a.basis.full_vals.to(device)
    mesh_b.basis.full_vals = mesh_b.basis.full_vals.to(device)

    mesh_a.laplacian._mass_matrix = mesh_a.laplacian._mass_matrix.to(device)
    mesh_b.laplacian._mass_matrix = mesh_b.laplacian._mass_matrix.to(device)
    mesh_b.basis.use_k = 20
    mesh_a.basis.use_k = 20

    p2p_to_fm = FmFromP2pConverter(pseudo_inverse=True)
    converter = P2pFromFmConverter(
        adjoint=False, neighbor_finder=GPUEuclideanNeighborFinder()
    )
    mesh_b.basis.use_k = 125
    mesh_a.basis.use_k = 125

    zoomout = ZoomOut(
        nit=20, step=5, p2p_from_fm_converter=converter, fm_from_p2p_converter=p2p_to_fm
    )
    fmap = zoomout(fmap, mesh_b.basis, mesh_a.basis)
    p2p = converter(fmap, mesh_b.basis, mesh_a.basis)
    p2p = p2p.cpu().numpy()

    elapsed_time = time.time() - start_time
    tqdm.write(f">>>> compute_p2p_fmaps_zoomout elapsed {elapsed_time:.4f}s")
    return p2p, elapsed_time


def compute_p2p_fmaps_neural_zoomout(
    source_path, target_path, source_input, target_input, device: str
):
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
        LBCommutativityEnforcing.from_bases(mesh_b.basis, mesh_a.basis, weight=1e-2),
        OperatorCommutativityEnforcing.from_multiplication(
            mesh_b.basis, descr_b, mesh_a.basis, descr_a, weight=1e-1
        ),
    ]

    objective = FactorSum(factors)
    x0 = np.zeros((mesh_a.basis.spectrum_size, mesh_b.basis.spectrum_size))
    optimizer = ScipyMinimize(method="L-BFGS-B")
    res = optimizer.minimize(objective, x0, fun_jac=objective.gradient)

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
    zoomout = ZoomOut(
        nit=20, step=5, p2p_from_fm_converter=converter, fm_from_p2p_converter=p2p_to_fm
    )
    fmap = zoomout(fmap, mesh_b.basis, mesh_a.basis)
    p2p = converter(fmap, mesh_b.basis, mesh_a.basis)
    p2p = p2p.cpu().numpy()

    elapsed_time = time.time() - start_time
    tqdm.write(f">>>> compute_p2p_fmaps_neural_zoomout elapsed {elapsed_time:.4f}s")
    return p2p, elapsed_time
