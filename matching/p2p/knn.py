"""KNN-based point-to-point matching methods, with optional ZoomOut refinement."""

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
from geomfum.laplacian import LaplacianFinder
from geomfum.refine import ZoomOut
from geomfum.shape import PointCloud, TriangleMesh
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


def compute_p2p_knn(source_input, target_input):
    tqdm.write("> computing p2p with knn")
    start_time = time.time()

    nbrs = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(
        target_input.cpu().numpy()
    )
    _, p2p = nbrs.kneighbors(source_input.cpu().numpy())
    p2p = p2p[:, 0]

    elapsed_time = time.time() - start_time
    tqdm.write(f">>>> compute_p2p_knn elapsed {elapsed_time:.4f}s")
    return p2p, elapsed_time


def compute_p2p_knn_zoomout(
    source_path, target_path, source_input, target_input, device: str
):
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
    zoomout = ZoomOut(
        nit=20, step=5, p2p_from_fm_converter=converter, fm_from_p2p_converter=p2p_to_fm
    )
    fmap = zoomout(fmap, mesh_b.basis, mesh_a.basis)

    p2p = converter(fmap, mesh_b.basis, mesh_a.basis)
    p2p = p2p.cpu().numpy()

    elapsed_time = time.time() - start_time
    tqdm.write(f">>>> compute_p2p_knn_zoomout elapsed {elapsed_time:.4f}s")
    return p2p, elapsed_time


def compute_p2p_knn_neural_zoomout(
    source_path, target_path, source_input, target_input, device: str
):
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
    zoomout = ZoomOut(
        nit=20, step=5, p2p_from_fm_converter=converter, fm_from_p2p_converter=p2p_to_fm
    )
    fmap = zoomout(fmap, mesh_b.basis, mesh_a.basis)
    p2p = converter(fmap, mesh_b.basis, mesh_a.basis)
    p2p = p2p.cpu().numpy()

    elapsed_time = time.time() - start_time
    tqdm.write(f">>>> compute_p2p_knn_neural_zoomout elapsed {elapsed_time:.4f}s")
    return p2p, elapsed_time
