import geomstats.backend as gs
import numpy as np
import potpourri3d as pp3d
import torch
import torch.nn.functional as F
import trimesh
from geomfum.laplacian import LaplacianFinder
from geomfum.metric import HeatDistanceMetric
from geomfum.shape.mesh import TriangleMesh
from geomfum.shape.point_cloud import PointCloud
from sklearn.neighbors import NearestNeighbors


def compute_geodesic_error(dists, p2p, corr_a=None, corr_b=None):
    """
    Compute geodesic distance error between ground truth and predicted correspondences.

    Args:
        dist: Geodesic distance matrix on the target shape
        p2p: Point-to-point correspondence map between reconstructed and target shape
        corr_a: Source correspondence indices
        corr_b: Target correspondence indices

    Returns:
        Average geodesic distance error
    """
    if corr_a is None or corr_b is None:
        p2p_gt = np.arange(len(p2p))
        return dists[p2p, p2p_gt].mean() / dists.max()
    else:
        return dists[p2p[corr_a], corr_b].mean() / dists.max()


def compute_dirichlet_energy(mesh, mesh_target, p2p):
    """
    Compute Dirichlet energy of the mapping.
    """
    # Dirichlet energy is defined as the sum of squared gradient magnitudes
    # For a discrete mesh, we can approximate it using the Laplacian

    # Create map from source to target vertices as vectors
    mapping = gs.array(mesh_target.vertices[p2p])

    if len(mesh.faces) > 0:
        mesh_gf = TriangleMesh(np.array(mesh.vertices), np.array(mesh.faces))
        L, A = mesh_gf.laplacian.find()

        # Compute Dirichlet energy
        energy_x = mapping[:, 0].T @ L @ mapping[:, 0]
        energy_y = mapping[:, 1].T @ L @ mapping[:, 1]
        energy_z = mapping[:, 2].T @ L @ mapping[:, 2]

        total_energy = energy_x + energy_y + energy_z
    else:
        mesh_gf = PointCloud(np.array(mesh.vertices))
        L, A = mesh_gf.laplacian.find(
            laplacian_finder=LaplacianFinder.from_registry(mesh=False, which="robust")
        )

        # Convert to CSR format for efficient matrix operations
        energy_x = mapping[:, 0].T @ L @ mapping[:, 0]
        energy_y = mapping[:, 1].T @ L @ mapping[:, 1]
        energy_z = mapping[:, 2].T @ L @ mapping[:, 2]

        total_energy = energy_x + energy_y + energy_z

    return total_energy / mesh_gf.n_vertices


def compute_coverage(p2p, source_vertices, target_vertices):
    """
    Ratio of unique target vertices covered by the mapping from source to
    target, adjusted by the cardinality ratio.
    """
    unique_targets = np.unique(p2p)
    cardinality_ratio = len(target_vertices) / len(source_vertices)
    coverage_ratio = (len(unique_targets) / len(target_vertices)) * cardinality_ratio

    return coverage_ratio
