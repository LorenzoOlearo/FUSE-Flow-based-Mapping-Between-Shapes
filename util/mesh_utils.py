"""Mesh utilities for features computation and sampling"""

import numpy as np
import torch
import trimesh
import os
import networkx as nx
import scipy
import torch
import fpsample

import scipy.sparse
import scipy.sparse.csgraph
import pandas as pd

import pandas as pd
import scipy
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import shortest_path, connected_components
from numpy.lib.format import open_memmap

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from sklearn import neighbors
from sklearn.manifold import MDS

# Add this to a new cell in your notebook
import numpy as np
from scipy.sparse.csgraph import shortest_path
from scipy.sparse import coo_matrix
from scipy.spatial.distance import cdist
import potpourri3d as pp3d

import geomfum
from geomfum.shape.mesh import TriangleMesh
from geomfum.shape.point_cloud import PointCloud

from geomfum.descriptor.spectral import (
    WaveKernelSignature,
    LandmarkWaveKernelSignature,
    HeatKernelSignature,
    LandmarkHeatKernelSignature,
)
from geomfum.descriptor.pipeline import DescriptorPipeline, ArangeSubsampler
from geomfum.laplacian import LaplacianFinder, LaplacianSpectrumFinder

from geomfum.shape.mesh import TriangleMesh
from geomfum.metric import HeatDistanceMetric
import potpourri3d as pp3d

from numpy.core._exceptions import _ArrayMemoryError


########################FUNCTIONS FOR MESH NORMALIZATION ##########################
def normalize_mesh_unit(mesh):
    """
    Normalize the mesh by centering and rescaling it.
    Args:
        mesh (trimesh.Trimesh): The input mesh to normalize.
    Returns:
        trimesh.Trimesh: The normalized mesh.
    """
    rescale = max(mesh.extents) / 2.0
    tform = [-(mesh.bounds[1][i] + mesh.bounds[0][i]) / 2.0 for i in range(3)]
    matrix = np.eye(4)
    matrix[:3, 3] = tform
    mesh.apply_transform(matrix)
    matrix = np.eye(4)
    matrix[:3, :3] /= rescale
    mesh.apply_transform(matrix)

    return mesh


def normalize_mesh_08(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    centroid = torch.tensor(mesh.centroid, dtype=torch.float32)
    verts = torch.tensor(mesh.vertices, dtype=torch.float32)

    verts -= centroid
    verts /= verts.abs().max()
    verts *= 0.8

    mesh.vertices = verts.numpy()

    return mesh


def pc_normalize(pc):
    """
    Normalize the point cloud by centering and scaling it.
    Args:
        pc (np.ndarray): The input point cloud of shape (N, 3).
    Returns:
        np.ndarray: The normalized point cloud.
    """
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


############################## FUNCTIONS FOR SAMPLING PRIOR DISTRIBUTIONS #############################


def sample_sphere_volume(radius, center, num_points, device="cuda"):
    """
    Sample points uniformly within a sphere of given radius and center.
    Args:
        radius (float): Radius of the sphere.
        center (tuple): Center of the sphere (x, y, z).
        num_points (int): Number of points to sample.
        device (str): Device to use for tensor operations ('cuda' or 'cpu').
    Returns:
        torch.Tensor: Sampled points of shape (num_points, 3).
    """
    center = torch.tensor(center, dtype=torch.float32, device=device)
    r = radius * torch.rand(num_points, device=device) ** (
        1 / 3
    )  # Sample radius with proper scaling for uniform distribution
    theta = torch.rand(num_points, device=device) * 2 * np.pi  # Azimuthal angle
    phi = torch.acos(1 - 2 * torch.rand(num_points, device=device))  # Polar angle

    # Convert to Cartesian coordinates
    x = r * torch.sin(phi) * torch.cos(theta)
    y = r * torch.sin(phi) * torch.sin(theta)
    z = r * torch.cos(phi)

    points = torch.stack((x, y, z), dim=1) + center

    return points


def sample_sphere_volume_multidimensional(radius, center, num_points, device="cuda"):
    """
    Sample points uniformly within a d-dimensional sphere of given radius and center.

    Args:
        radius (float): Radius of the sphere.
        center (tuple or list): Center of the sphere (length = dimension d).
        num_points (int): Number of points to sample.
        device (str): Device for tensor operations.

    Returns:
        torch.Tensor: Sampled points of shape (num_points, d).
    """
    center = torch.tensor(center, dtype=torch.float32, device=device)
    d = len(center)  # Dimension inferred from center

    # Sample radius with proper scaling for uniform distribution in d dimensions
    r = radius * torch.rand(num_points, device=device) ** (1 / d)

    # Sample directions by normalizing Gaussian vectors
    directions = torch.randn(num_points, d, device=device)
    directions = directions / directions.norm(dim=1, keepdim=True)

    points = directions * r.view(-1, 1) + center
    return points


def sample_initial_distribution(num_points: int, embedding_dim: int, distribution: str, device: str):
    """
    Generate random samples based on the specified distribution.
    Args:
        args (argparse.Namespace): Command-line arguments containing distribution type and other parameters.
        device (str): Device to use for tensor operations ('cuda' or 'cpu').
    Returns:
        torch.Tensor: Generated samples of shape (num_points, embedding_dim).
    """
    center = torch.tensor(center, dtype=torch.float32, device=device)
    d = len(center)  # Dimensionality inferred from center

    # Sample Gaussian vectors
    directions = torch.randn(num_points, d, device=device)

    # Normalize to get points on the unit sphere
    directions = directions / directions.norm(dim=1, keepdim=True)

    # Scale by radius and shift by center
    points = directions * radius + center

    return points


def sample_cube_multidimensional(side_length, center, num_points, device="cuda"):
    """
    Sample uniformly from a d-dimensional cube.

    The cube ranges from:
        center[i] - side_length/2  to  center[i] + side_length/2
    """

    center = torch.as_tensor(center, dtype=torch.float32, device=device)
    d = len(center)

    half = side_length / 2.0

    # Uniform in [-half, half]^d
    offsets = (torch.rand(num_points, d, device=device) - 0.5) * (2 * half)

    return center + offsets


def sample_fitted_gaussian(num_points: int, target_data: torch.Tensor, device: str):
    """
    Sample points from a Gaussian distribution fitted to the target data.
    Args:
        num_points (int): Number of points to sample.
        target_data (torch.Tensor): Target data to fit the Gaussian to. Shape (N, embedding_dim).
        device (str): Device for tensor operations.
    Returns:
        torch.Tensor: Sampled points of shape (num_points, embedding_dim).
    """
    mean = torch.mean(target_data, dim=0)
    cov = torch.from_numpy(np.cov(target_data.cpu().numpy(), rowvar=False)).to(device)

    # Sample from multivariate normal distribution
    samples = torch.distributions.MultivariateNormal(
        mean, covariance_matrix=cov
    ).sample((num_points,))

    return samples


def sample_initial_distribution(
    num_points: int,
    embedding_dim: int,
    distribution: str,
    device: str,
    target_data: torch.Tensor = None,
):
    """
    Sample points from the specified initial distribution.
    Args:
        num_points (int): Number of points to sample.
        embedding_dim (int): Dimensionality of the embedding space.
        distribution (str): Type of distribution to sample from. One of {"gaussian", "fitted_gaussian", "sphere", "cube"}.
        device (str): Device for tensor operations.
        target_data (torch.Tensor, optional): Target data for fitted Gaussian distribution. Shape (N, embedding_dim).
    Returns:
        torch.Tensor: Sampled points of shape (num_points, embedding_dim).
    """

    center = (0,) * embedding_dim

    if distribution == "gaussian":
        samples = torch.randn(num_points, embedding_dim).to(device)

    elif distribution == "fitted_gaussian":
        if target_data is None:
            raise ValueError(
                "target_data must be provided for fitted_gaussian distribution."
            )
        samples = sample_fitted_gaussian(
            num_points=num_points,
            target_data=target_data,
            device=device,
        )

    elif distribution == "sphere":
        samples = sample_sphere_volume_multidimensional(
            radius=1,
            center=(0.5, 0.5, 0.5, 0.5, 0.5),
            num_points=num_points,
            device=device,
        )
    else:
        raise ValueError(f"Unsupported distribution type: {distribution}")

    return samples


############################### FUNCTIONS FOR MESH EMBEDDINGS #############################
def generate_embeddings(
    mesh, embedding_type, num_points, features, device: str
) -> torch.Tensor:
    """
    Generate point embeddings from a mesh using the specified embedding strategy.

    Depending on the `embedding_type` option, this function returns either:
      - Vertex coordinates only ("xyz")
      - Features only ("features_only")
      - Concatenated coordinates + features ("features")

    If the mesh contains faces, embeddings are sampled from the surface;
    otherwise, embeddings are sampled from vertices.

    Args:
        - mesh (trimesh.Trimesh): The input 3D mesh.
        - features (torch.Tensor | np.ndarray): Per-vertex features aligned with `mesh.vertices`.
        - embedding_type (str): One of {"xyz", "features", "features_only"}.
        - num_points_ (int): Number of points to sample.
        - device (str): Target device for returned embeddings

    Returns:
        torch.Tensor: Tensor of shape (num_points, embedding_dim).
    """

    if len(mesh.faces) > 0:
        if embedding_type in {"features", "features_only"}:
            # Interpolated features sampled across faces
            embedding = get_interpolated_feats(
                mesh, features, num_points, device=device
            )

            if embedding_type == "features_only":
                # Drop XYZ coords, keep only feature channels
                embedding = embedding[:, 3:]

        elif embedding_type == "xyz":
            # Sample 3D points directly from mesh surface
            samples, _ = trimesh.sample.sample_surface_even(mesh, num_points)
            embedding = torch.tensor(samples, dtype=torch.float32, device=device)

        else:
            raise ValueError(f"Unsupported embedding type: {embedding_type}")

    else:
        indices = np.random.choice(len(mesh.vertices), num_points, replace=True)

        if embedding_type == "xyz":
            samples = mesh.vertices[indices]
            embedding = torch.tensor(samples, dtype=torch.float32, device=device)

        elif embedding_type == "features_only":
            embedding = features[indices]

        elif embedding_type == "features":
            # Concatenate XYZ coordinates with features
            samples = torch.tensor(mesh.vertices, dtype=torch.float32, device=device)
            feat_tensor = (
                features
                if torch.is_tensor(features)
                else torch.tensor(features, dtype=torch.float32, device=device)
            )
            full_embedding = torch.cat([samples, feat_tensor], dim=-1)
            embedding = full_embedding[indices]

        else:
            raise ValueError(f"Unsupported embedding type: {embedding_type}")

    return embedding


def get_interpolated_feats(mesh, features, num_points, device):
    """
    Interpolates features to sampled points on the mesh surface.

    Parameters:
    -----------
    mesh : trimesh.Trimesh
        The input mesh.
    args : argparse.Namespace
        Command-line arguments containing features and other parameters.
    device : torch.device
        Device to use for tensor operations ('cuda' or 'cpu').
    Returns:
    --------
    torch.Tensor
        Tensor of shape (num_points, embedding_dim) containing sampled points and their interpolated features.
    """

    # Sample points
    # samples, face_indices = trimesh.sample.sample_surface_even(mesh, num_points)
    # samples, face_indices = trimesh.sample.sample_surface(mesh, num_points, face_weight=mesh.area)
    samples, face_indices = trimesh.sample.sample_surface(mesh, num_points)
    print(
        f"[OCIO] Sampled {len(samples)} points on mesh surface for feature interpolation."
    )
    samples = samples.astype(np.float32)

    # Convert to tensors
    samples_tensor = torch.tensor(samples, device=device).float()
    faces = mesh.faces[face_indices]

    # Get all vertices for each face (batch processing)
    face_vertices = mesh.vertices[faces]  # Shape: [num_points, 3, 3]

    # Compute face normals (vectorized)
    v0 = face_vertices[:, 1] - face_vertices[:, 0]  # Shape: [num_points, 3]
    v1 = face_vertices[:, 2] - face_vertices[:, 0]  # Shape: [num_points, 3]

    # Calculate vectors for barycentric coordinates
    v2 = samples - face_vertices[:, 0]  # Shape: [num_points, 3]

    # Compute dot products for barycentric calculation
    d00 = np.sum(v0 * v0, axis=1)  # Shape: [num_points]
    d01 = np.sum(v0 * v1, axis=1)  # Shape: [num_points]
    d11 = np.sum(v1 * v1, axis=1)  # Shape: [num_points]
    d20 = np.sum(v2 * v0, axis=1)  # Shape: [num_points]
    d21 = np.sum(v2 * v1, axis=1)  # Shape: [num_points]

    # Calculate denominator
    denom = d00 * d11 - d01 * d01  # Shape: [num_points]

    # Handle barycentric coordinates calculation
    valid_mask = np.abs(denom) >= 1e-10

    # Initialize barycentric coordinates
    v = np.zeros(len(samples))
    w = np.zeros(len(samples))

    # Calculate barycentric coordinates for valid triangles
    v[valid_mask] = (
        d11[valid_mask] * d20[valid_mask] - d01[valid_mask] * d21[valid_mask]
    ) / denom[valid_mask]
    w[valid_mask] = (
        d00[valid_mask] * d21[valid_mask] - d01[valid_mask] * d20[valid_mask]
    ) / denom[valid_mask]
    u = 1.0 - v - w

    # Clamp and normalize barycentric coordinates
    u = np.clip(u, 0.0, 1.0)
    v = np.clip(v, 0.0, 1.0)
    w = np.clip(w, 0.0, 1.0)

    total = u + v + w
    u = u / total
    v = v / total
    w = w / total

    # Convert everything to PyTorch for faster GPU computation
    u_tensor = (
        torch.tensor(u, device=device).float().unsqueeze(1)
    )  # Shape: [num_points, 1]
    v_tensor = (
        torch.tensor(v, device=device).float().unsqueeze(1)
    )  # Shape: [num_points, 1]
    w_tensor = (
        torch.tensor(w, device=device).float().unsqueeze(1)
    )  # Shape: [num_points, 1]
    valid_mask_tensor = torch.tensor(valid_mask, device=device)

    # Convert face indices to PyTorch tensors
    face_idx_tensor = torch.tensor(faces, device=device)  # Shape: [num_points, 3]

    # Get landmark geodesic distances for all vertices in all faces
    # Shape: [num_points, 3, num_landmarks]
    vertex_feats = features[face_idx_tensor]

    # Interpolate using barycentric coordinates
    # Multiply each vertex's geodesic distances by its barycentric weight
    # Shape after multiplication: [num_points, num_landmarks]
    interpolated_feats = (
        u_tensor * vertex_feats[:, 0]  # First vertex contribution
        + v_tensor * vertex_feats[:, 1]  # Second vertex contribution
        + w_tensor * vertex_feats[:, 2]  # Third vertex contribution
    )

    # For invalid triangles, use nearest vertex
    if not torch.all(valid_mask_tensor):
        # Calculate squared distances to each vertex for all points
        # Shape: [num_points, 3]
        vertex_distances = torch.sum(
            (torch.tensor(face_vertices, device=device) - samples_tensor.unsqueeze(1))
            ** 2,
            dim=2,
        )

        # Find nearest vertex index (0, 1, or 2) for each face
        # Shape: [num_points]
        nearest_vertex_idx = torch.argmin(vertex_distances, dim=1)

        # Create indices for gathering
        point_indices = torch.arange(len(samples), device=device)

        # Get the actual vertex indices from the faces
        # Shape: [num_points]
        selected_vertices = face_idx_tensor[point_indices, nearest_vertex_idx]

        # Get geodesic distances for nearest vertices
        # Shape: [num_points, num_landmarks]
        nearest_geodesic = features[selected_vertices]

        # Use nearest vertex distances for invalid triangles
        invalid_mask_tensor = ~valid_mask_tensor
        interpolated_feats[invalid_mask_tensor] = nearest_geodesic[invalid_mask_tensor]

    return torch.cat([samples_tensor, interpolated_feats], -1)


def get_shape_diameter(
    mesh: trimesh.Trimesh,
    target: str,
    dists_path: str,
    recompute: bool = False,
) -> float:
    """
    Compute or load geodesic distance matrix for a target mesh and return the shape diameter (max distance).

    Args:
        mesh (trimesh.Trimesh): The mesh to compute distances on.
        target (str): Identifier for caching (e.g. filename stem).
        dists_path (str): Directory path where the distance matrix is cached.
        recompute (bool): If True, recompute and overwrite the cached distances even if present.

    Returns:
        float: Shape diameter, i.e., the maximum geodesic distance.
    """
    os.makedirs(dists_path, exist_ok=True)
    dist_cache_path = os.path.join(dists_path, f"{target}_dists.npy")

    dist = None
    if not recompute and os.path.exists(dist_cache_path):
        try:
            dist = np.load(dist_cache_path)
        except Exception:
            dist = None

    if dist is None:
        if mesh.is_watertight and len(mesh.faces) > 0:
            print(f"Computing geodesic distances for {target} using GeomFum...")
            mesh_gf = TriangleMesh(mesh.vertices, np.array(mesh.faces))
            heat = HeatDistanceMetric.from_registry(mesh_gf)
            dist = heat.dist_matrix()
            np.save(dist_cache_path, dist)
            shape_diameter = float(np.max(dist))
        else:
            print(
                f"Computing approximate geodesic distances for {target} using potpourri3d..."
            )
            solver = pp3d.PointCloudHeatSolver(mesh.vertices)
            max_dist = 0.0
            for idx in range(len(mesh.vertices)):
                d = solver.compute_distance(idx)
                max_dist = max(max_dist, np.max(d))
            shape_diameter = float(max_dist)
            np.save(dist_cache_path, shape_diameter)
    else:
        shape_diameter = float(np.max(dist)) if dist.ndim > 0 else float(dist)

    return shape_diameter


def mesh_geodesics(
    mesh: trimesh.Trimesh,
    target: str,
    recompute: bool,
    dists_path: str,
    n_start: int = 50,
    largest_component_only: bool = False,
) -> tuple[np.ndarray | None, float]:
    """
    Compute or load the geodesic distance matrix and diameter of a mesh
    using Dijkstra's algorithm, with caching, low-memory fallback, and
    optional restriction to the largest connected component.
    """

    os.makedirs(dists_path, exist_ok=True)
    dists_file = os.path.join(dists_path, f"{target}_dists.npy")
    diameters_csv = os.path.join(dists_path, "diameters.csv")

    edges, lengths = mesh.edges_unique, mesh.edges_unique_length
    n_vertices = len(mesh.vertices)

    # --- Build adjacency matrix once ---
    adjacency = scipy.sparse.csr_matrix(
        (lengths, (edges[:, 0], edges[:, 1])),
        shape=(n_vertices, n_vertices),
    )
    adjacency = adjacency + adjacency.T  # ensure undirected

    # --- Restrict to largest connected component if requested ---
    if largest_component_only:
        print(
            f"[DISTS INFO] Restricting computation to the largest connected component of '{target}'."
        )
        n_components, labels = connected_components(csgraph=adjacency, directed=False)
        sizes = np.bincount(labels)
        largest_label = np.argmax(sizes)
        mask = labels == largest_label

        vertex_indices = np.where(mask)[0]
        adjacency = adjacency[mask][:, mask]
        mesh_vertices = mesh.vertices[mask]
        n_vertices = len(mesh_vertices)

        print(
            f"[DISTS INFO] Largest component has {n_vertices} vertices "
            f"({sizes[largest_label]} / {len(mesh.vertices)} total)."
        )
    else:
        mesh_vertices = mesh.vertices

    try:
        # --- Try loading cached matrix if available ---
        if not recompute and os.path.exists(dists_file):
            print(
                f"[DISTS] Attempting to load cached geodesic distances for '{target}' (lazy memmap)..."
            )
            try:
                # Lazy load: do not read full matrix into RAM
                shape_dists = np.load(dists_file, mmap_mode="r")
                print(
                    f"[DISTS INFO] Successfully memory-mapped cached distances for '{target}'."
                )
            except MemoryError:
                print(
                    f"[DISTS WARN] MemoryError while loading cached matrix for '{target}'. "
                    f"Falling back to recomputation..."
                )
                raise  # trigger fallback below

        else:
            print(
                f"[DISTS] Computing full geodesic distance matrix for '{target}' (Dijkstra)..."
            )
            shape_dists = shortest_path(
                csgraph=adjacency, directed=False, return_predecessors=False, method="D"
            )

            # Handle disconnected components
            if np.isinf(shape_dists).any():
                print(
                    f"[DISTS WARN] Disconnected components detected in '{target}'. "
                    f"Replacing inf geodesic distances with Euclidean distances."
                )
                eucl_dists = cdist(mesh_vertices, mesh_vertices)
                inf_mask = np.isinf(shape_dists)
                shape_dists[inf_mask] = eucl_dists[inf_mask]

            np.save(dists_file, shape_dists)
            print(f"[DISTS INFO] Saved geodesic distances to '{dists_file}'.")

        finite_dists = shape_dists[np.isfinite(shape_dists)]
        diameter = float(finite_dists.max()) if finite_dists.size else 0.0

    except (MemoryError, np.core._exceptions._ArrayMemoryError):
        # --- Low-memory fallback: streamed Dijkstra computation ---
        print(
            f"[DISTS WARN] MemoryError computing or loading full matrix for '{target}'."
        )
        print(
            f"[DISTS INFO] Falling back to incremental vertex-by-vertex Dijkstra computation "
            f"and on-disk concatenation..."
        )

        partial_file = os.path.join(dists_path, f"{target}_dists_partial.npy")

        shape_dists = open_memmap(
            partial_file, mode="w+", dtype=np.float64, shape=(n_vertices, n_vertices)
        )

        for i in range(n_vertices):
            if i % 100 == 0:
                print(f"[DISTS INFO] Processing vertex {i}/{n_vertices}...")
            try:
                d_i = shortest_path(
                    csgraph=adjacency, directed=False, indices=i, method="D"
                )
            except MemoryError:
                print(
                    f"[DISTS ERROR] MemoryError even for single vertex {i}. Aborting streamed computation."
                )
                del shape_dists
                os.remove(partial_file)
                shape_dists = None
                break

            shape_dists[i, :] = d_i  # write directly to disk

        if shape_dists is not None:
            print(
                f"[DISTS INFO] Finished streamed Dijkstra computation for '{target}'."
            )
            print(
                f"[DISTS INFO] Saved partial geodesic distance matrix to '{partial_file}'."
            )

            # Handle disconnected components
            if np.isinf(shape_dists).any():
                print(
                    f"[DISTS WARN] Disconnected components detected in '{target}'. "
                    f"Replacing inf geodesic distances with Euclidean distances (streamed mode)."
                )
                eucl_dists = cdist(mesh_vertices, mesh_vertices)
                inf_mask = np.isinf(shape_dists)
                shape_dists[inf_mask] = eucl_dists[inf_mask]

            final_file = os.path.join(dists_path, f"{target}_dists.npy")
            os.replace(partial_file, final_file)
            print(f"[DISTS INFO] Final geodesic distances saved to '{final_file}'.")

            finite_dists = shape_dists[np.isfinite(shape_dists)]
            diameter = float(finite_dists.max()) if finite_dists.size else 0.0

        else:
            print(f"[DISTS WARN] Falling back to two-sweep diameter approximation.")
            max_dists = []
            start_indices = np.random.choice(
                n_vertices, size=min(n_start, n_vertices), replace=False
            )
            for idx in start_indices:
                d_from_start = shortest_path(
                    csgraph=adjacency, directed=False, indices=idx, method="D"
                )
                farthest_idx = np.nanargmax(d_from_start)
                d_from_far = shortest_path(
                    csgraph=adjacency, directed=False, indices=farthest_idx, method="D"
                )
                max_dists.append(np.nanmax(d_from_far))
            diameter = float(np.max(max_dists))
            shape_dists = None

    # --- Save or update diameter record ---
    if os.path.exists(diameters_csv):
        df = pd.read_csv(diameters_csv)
    else:
        df = pd.DataFrame(columns=["target", "diameter"])

    if target in df["target"].values:
        if recompute:
            df.loc[df["target"] == target, "diameter"] = diameter
            print(f"[DISTS INFO] Updated diameter entry for '{target}'.")
    else:
        df = pd.concat(
            [df, pd.DataFrame({"target": [target], "diameter": [diameter]})],
            ignore_index=True,
        )

    df.to_csv(diameters_csv, index=False)
    print(f"[DISTS INFO] Diameter for '{target}': {diameter:.4f}")

    return shape_dists, diameter


def mesh_geodesics_heat_method(
    mesh: trimesh.Trimesh,
    target: str,
    recompute: bool,
    dists_path: str,
) -> tuple[np.ndarray, float]:
    """
    Compute or load the geodesic distance matrix and diameter of a mesh
    using Heat Method, with caching support.

    Args:
        mesh: Trimesh object representing the mesh geometry.
        target: Unique identifier for the mesh (e.g., filename stem).
        recompute: If True, forces recomputation even if cached results exist.
        dists_path: Directory to cache/load distance matrices and diameter CSV.
    Returns:
        (shape_dists, diameter):
            shape_dists: (N, N) ndarray of geodesic distances between vertices.
            diameter: Float, maximum finite distance across the mesh.
    """
    os.makedirs(dists_path, exist_ok=True)
    dists_file = os.path.join(dists_path, f"{target}_dists.npy")
    diameters_csv = os.path.join(dists_path, "diameters.csv")

    # --- Load or compute full NxN distance matrix ---
    if not recompute and os.path.exists(dists_file):
        print(f"[DISTS] Loading cached geodesic distances for '{target}'.")
        shape_dists = np.load(dists_file)
    else:
        print(f"[DISTS] Computing geodesic distances for '{target}' (Heat Method)...")

        mesh_gf = TriangleMesh(mesh.vertices, np.array(mesh.faces))
        heat = HeatDistanceMetric.from_registry(mesh_gf)
        shape_dists = heat.dist_matrix()

        np.save(dists_file, shape_dists)

    # --- Compute geodesic diameter ---
    finite_dists = shape_dists[np.isfinite(shape_dists)]
    diameter = float(finite_dists.max()) if finite_dists.size else 0.0

    # --- Save or update diameter record ---
    if os.path.exists(diameters_csv):
        df = pd.read_csv(diameters_csv)
    else:
        df = pd.DataFrame(columns=["target", "diameter"])

    if target in df["target"].values:
        if recompute:
            df.loc[df["target"] == target, "diameter"] = diameter
            print(f"[DISTS] Updated diameter entry for '{target}'.")
    else:
        df = pd.concat(
            [df, pd.DataFrame({"target": [target], "diameter": [diameter]})],
            ignore_index=True,
        )

    df.to_csv(diameters_csv, index=False)

    return shape_dists, diameter


def pointcloud_geodesics(
    pt: trimesh.Trimesh,
    target: str,
    recompute: bool,
    dists_path: str,
    n_start: int = 5,
) -> tuple[np.ndarray | None, float]:
    """
    Compute or load the geodesic distance matrix and diameter of a point cloud
    using the Heat Method, with caching and low-memory fallback.

    Args:
        pt: Trimesh object representing the point cloud geometry.
        target: Unique identifier for the point cloud (e.g., filename stem).
        recompute: If True, forces recomputation even if cached results exist.
        dists_path: Directory to cache/load distance matrices and diameter CSV.
        n_start: Number of random start points used for fallback diameter approximation.
    Returns:
        (shape_dists, diameter):
            shape_dists: (N, N) ndarray of geodesic distances, or None if approximated.
            diameter: Float, maximum finite distance across the point cloud.
    """

    os.makedirs(dists_path, exist_ok=True)
    dists_file = os.path.join(dists_path, f"{target}_dists.npy")
    diameters_csv = os.path.join(dists_path, "diameters.csv")

    solver = pp3d.PointCloudHeatSolver(pt.vertices)
    n_points = len(pt.vertices)

    # --- Try full NxN computation ---
    try:
        if not recompute and os.path.exists(dists_file):
            print(f"[DISTS] Loading cached geodesic distances for '{target}'.")
            shape_dists = np.load(dists_file)
        else:
            print(
                f"[DISTS] Computing full geodesic distance matrix for '{target}' (Heat Method)..."
            )
            shape_dists = np.zeros((n_points, n_points), dtype=np.float32)

            for idx in range(n_points):
                shape_dists[idx, :] = solver.compute_distance(idx)

            np.save(dists_file, shape_dists)

        finite_dists = shape_dists[np.isfinite(shape_dists)]
        diameter = float(finite_dists.max()) if finite_dists.size else 0.0

    except _ArrayMemoryError:
        # --- Low-memory fallback: approximate diameter in two-sweep fashion ---
        print(
            f"[DISTS WARN] MemoryError computing full matrix for '{target}'. Falling back to two-sweep approximation."
        )

        max_dists = []
        start_indices = np.random.choice(
            n_points, size=min(n_start, n_points), replace=False
        )

        for idx in start_indices:
            # First sweep
            d_from_start = solver.compute_distance(idx)
            farthest_idx = np.nanargmax(d_from_start)

            # Second sweep
            d_from_far = solver.compute_distance(farthest_idx)
            max_dists.append(np.nanmax(d_from_far))

        diameter = float(np.max(max_dists))
        shape_dists = None

    # --- Save or update diameter record ---
    if os.path.exists(diameters_csv):
        df = pd.read_csv(diameters_csv)
    else:
        df = pd.DataFrame(columns=["target", "diameter"])

    if target in df["target"].values:
        if recompute:
            df.loc[df["target"] == target, "diameter"] = diameter
            print(f"[DISTS] Updated diameter entry for '{target}'.")
    else:
        df = pd.concat(
            [df, pd.DataFrame({"target": [target], "diameter": [diameter]})],
            ignore_index=True,
        )

    df.to_csv(diameters_csv, index=False)

    return shape_dists, diameter


######################## # FUNCTIONS FOR COMPUTING FEATURES ########################


def compute_features(mesh, args, device):
    """
    Compute geodesic distances from each vertex to the landmarks, return them normalized on axis 0.

    Parameters:
    -----------
    mesh : trimesh.Trimesh
        The input mesh
    lm : numpy.ndarray
        Array of landmark vertex indices

    Returns:
    --------
    torch.Tensor
        Tensor of shape (num_vertices, len(lm)) containing geodesic distances
        from each vertex to each landmark vertex, normalized along axis 0.
    """

    features = None

    if args.embedding_type == "xyz":
        return torch.tensor(mesh.vertices, device=device).float()
    if args.features_type == "landmarks":
        if len(mesh.faces) > 0:
            if args.use_heat_method:
                print("[FEATURES] Using Heat Method for geodesic distances computation")
                features = torch.tensor(
                    compute_geodesic_distances_heat_method(
                        mesh, source_index=np.array(args.landmarks)
                    )
                ).T.to(device)
            else:
                print("[FEATURES] Using Dijkstra for geodesic distances computation")
                features = torch.tensor(
                    compute_geodesic_distances(
                        mesh, source_index=np.array(args.landmarks)
                    )
                ).T.to(device)
        else:
            features = torch.tensor(
                compute_geodesic_distances_pointcloud(
                    mesh, source_index=np.array(args.landmarks)
                )
            ).T.to(device)
    elif args.features_type == "landmarks_exp":
        if len(mesh.faces) > 0:
            features = torch.exp(
                -torch.tensor(
                    compute_geodesic_distances(
                        mesh, source_index=np.array(args.landmarks)
                    )
                ).T.to(device)
            )
        else:
            features = torch.exp(
                -torch.tensor(
                    compute_geodesic_distances_pointcloud(
                        mesh, source_index=np.array(args.landmarks)
                    )
                ).T.to(device)
            )
    elif args.features_type == "landmarks_biharmonic":
        features = torch.tensor(
            compute_biharmonic_distances(mesh, source_index=np.array(args.landmarks))
        ).T.to(device)
    elif args.features_type == "wks_5":
        features = torch.tensor(compute_wks(mesh, num_desc=5)).to(device)
    elif args.features_type == "wks_10":
        features = torch.tensor(compute_wks(mesh, num_desc=10)).to(device)
    elif args.features_type == "wks_20":
        features = torch.tensor(compute_wks(mesh, num_desc=20)).to(device)
    elif args.features_type == "wks_40":
        features = torch.tensor(compute_wks(mesh, num_desc=40)).to(device)
    elif args.features_type == "hks_5":
        features = torch.tensor(compute_wks(mesh, num_desc=5, hks=True)).to(device)
    elif args.features_type == "hks_10":
        features = torch.tensor(compute_wks(mesh, num_desc=10, hks=True)).to(device)
    elif args.features_type == "hks_20":
        features = torch.tensor(compute_wks(mesh, num_desc=20, hks=True)).to(device)
    elif args.features_type == "hks_40":
        features = torch.tensor(compute_wks(mesh, num_desc=40, hks=True)).to(device)
    elif args.features_type == "hks_plus_ldmk_5":
        features = torch.tensor(
            compute_wks(
                mesh, num_desc=5, hks=True, ldmk=True, indices=np.array(args.landmarks)
            )
        ).to(device)
    elif args.features_type == "hks_plus_ldmk_10":
        features = torch.tensor(
            compute_wks(
                mesh, num_desc=10, hks=True, ldmk=True, indices=np.array(args.landmarks)
            )
        ).to(device)
    elif args.features_type == "hks_plus_ldmk_20":
        features = torch.tensor(
            compute_wks(
                mesh, num_desc=20, hks=True, ldmk=True, indices=np.array(args.landmarks)
            )
        ).to(device)
    elif args.features_type == "hks_plus_ldmk_40":
        features = torch.tensor(
            compute_wks(
                mesh, num_desc=40, hks=True, ldmk=True, indices=np.array(args.landmarks)
            )
        ).to(device)
    elif args.features_type == "wks_plus_ldmk_5":
        features = torch.tensor(
            compute_wks(mesh, num_desc=5, ldmk=True, indices=np.array(args.landmarks))
        ).to(device)
    elif args.features_type == "wks_plus_ldmk_10":
        features = torch.tensor(
            compute_wks(mesh, num_desc=10, ldmk=True, indices=np.array(args.landmarks))
        ).to(device)
    elif args.features_type == "wks_plus_ldmk_20":
        features = torch.tensor(
            compute_wks(mesh, num_desc=20, ldmk=True, indices=np.array(args.landmarks))
        ).to(device)
    elif args.features_type == "wks_plus_ldmk_40":
        features = torch.tensor(
            compute_wks(mesh, num_desc=40, ldmk=True, indices=np.array(args.landmarks))
        ).to(device)

    elif args.features_type == "wks_plus_ldmk":
        if len(mesh.faces) > 0:
            ldmk = torch.tensor(
                compute_geodesic_distances(mesh, source_index=np.array(args.landmarks))
            ).T.to(device)
            geodesic = torch.tensor(compute_wks(mesh)).to(device)
        else:
            ldmk = torch.tensor(
                compute_geodesic_distances_pointcloud(
                    mesh, source_index=np.array(args.landmarks)
                )
            ).T.to(device)
            geodesic = torch.tensor(compute_wks(mesh)).to(device)
        features = torch.cat([geodesic, ldmk], -1)
    elif args.features_type == "wks_plus_ldmk_exp":
        if len(mesh.faces) > 0:
            ldmk = torch.tensor(
                compute_geodesic_distances(mesh, source_index=np.array(args.landmarks))
            ).T.to(device)
            geodesic = torch.exp(-torch.tensor(compute_wks(mesh)).to(device))
        else:
            ldmk = torch.tensor(
                compute_geodesic_distances_pointcloud(
                    mesh, source_index=np.array(args.landmarks)
                )
            ).T.to(device)
            geodesic = torch.exp(-torch.tensor(compute_wks(mesh)).to(device))
        features = torch.cat([geodesic, ldmk], -1)
    elif args.features_type == "mds":
        print("Computing MDS embedding")
        features = torch.tensor(
            compute_mds(mesh.vertices, mesh.faces, k=args.embedding_type_dim - 3)
        ).to(device)

    return features


def compute_geodesic_distmat(verts, faces):
    """
    Compute geodesic distance matrix using Dijkstra algorithm

    Args:
        verts (np.ndarray): array of vertices coordinates [n, 3]
        faces (np.ndarray): array of triangular faces [m, 3]

    Returns:
        geo_dist: geodesic distance matrix [n, n]
    """

    NN = 500

    # get adjacency matrix
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    vertex_adjacency = mesh.vertex_adjacency_graph
    assert nx.is_connected(vertex_adjacency), "Graph not connected"
    vertex_adjacency_matrix = nx.adjacency_matrix(
        vertex_adjacency, range(verts.shape[0])
    )
    # get adjacency distance matrix
    graph_x_csr = neighbors.kneighbors_graph(
        verts, n_neighbors=NN, mode="distance", include_self=False
    )
    distance_adj = csr_matrix((verts.shape[0], verts.shape[0])).tolil()
    distance_adj[vertex_adjacency_matrix != 0] = graph_x_csr[
        vertex_adjacency_matrix != 0
    ]
    # compute geodesic matrix
    geodesic_x = shortest_path(distance_adj, directed=False)
    if np.any(np.isinf(geodesic_x)):
        print("Inf number in geodesic distance. Increase NN.")
    return geodesic_x


def compute_mds(verts, faces, k=8):
    """
    Compute MDS embedding of the mesh using geodesic distance matrix
    Args:
        verts (np.ndarray): array of vertices coordinates [n, 3]
        faces (np.ndarray): array of triangular faces [m, 3]
        k (int): number of dimensions for MDS embedding
    Returns:
        embedding (np.ndarray): MDS embedding of the mesh [n, k]
    """
    D = compute_geodesic_distmat(verts, faces)
    mds = MDS(n_components=k, dissimilarity="precomputed")
    embedding = mds.fit_transform(D)

    return embedding


def compute_geodesic_distances_pointcloud(mesh, source_index):
    """
    Compute geodesic distances from landmarks to all points in the point cloud.
    Parameters:
    -----------
    mesh : trimesh.Trimesh
        The input mesh (point cloud).
    source_index : int
        Index of the source vertex (landmark).
    Returns:
    --------
    distances : np.ndarray
        Array of geodesic distances from the source vertex to all other vertices.
    """

    # Create a PointCloudHeatSolver instance
    solver = pp3d.PointCloudHeatSolver(np.array(mesh.vertices))

    distances = []
    source_index = [int(index) for index in source_index]
    for idx in source_index:
        distances.append([solver.compute_distance(int(idx))])

    return np.concatenate(distances, axis=0)


def compute_geodesic_distances(mesh, source_index):
    """
    Compute geodesic distances from a source vertex to all other vertices.
    If the mesh is disconnected, replaces infinite distances with Euclidean ones.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        The mesh on which to compute geodesic distances.
    source_index : int or sequence
        Index or sequence of source vertex indices.

    Returns
    -------
    distances : np.ndarray
        Array of distances; shape is (S, N) where S = number of sources and
        N = number of vertices. If a single source is provided it will still
        be returned as a 2D array with shape (1, N).
    """

    # --- Build adjacency ---
    edges = mesh.edges_unique
    edge_lengths = np.linalg.norm(
        mesh.vertices[edges[:, 0]] - mesh.vertices[edges[:, 1]], axis=1
    )
    n_vertices = len(mesh.vertices)

    graph = coo_matrix(
        (edge_lengths, (edges[:, 0], edges[:, 1])), shape=(n_vertices, n_vertices)
    )
    graph = graph + graph.T  # ensure undirected

    # --- Compute shortest paths ---
    # We don't need predecessors here so disable them to avoid unused variable warnings.
    distances = shortest_path(
        csgraph=graph, directed=False, indices=source_index, return_predecessors=False
    )

    # Ensure distances is 2D: shape (S, N)
    if distances.ndim == 1:
        distances = distances[np.newaxis, :]

    # --- Handle disconnected components ---
    if np.isinf(distances).any():
        print(
            f"[WARN] Disconnected components detected when computing from vertex {source_index}. "
            f"Replacing inf geodesic distances with Euclidean distances."
        )
        # Normalize source_index to integer array
        src_idx = np.atleast_1d(source_index).astype(int)

        verts = np.asarray(mesh.vertices)  # shape (N, 3)
        src_coords = verts[src_idx]  # shape (S, 3)

        # Compute Euclidean distances from each source to all vertices -> shape (S, N)
        # verts[None, :, :] -> (1, N, 3), src_coords[:, None, :] -> (S, 1, 3)
        eucl_dists = np.linalg.norm(verts[None, :, :] - src_coords[:, None, :], axis=2)

        inf_mask = np.isinf(distances)
        # Replace only the infinite entries with Euclidean estimates
        distances[inf_mask] = eucl_dists[inf_mask]

    return distances


def compute_geodesic_distances_heat_method(mesh, source_index):
    """
    Compute geodesic distances from landmarks to all points in the mesh using Heat Method.
    Parameters:
    -----------
    mesh : trimesh.Trimesh
        The input mesh.
    source_index : int
        Index of the source vertex (landmark).
    Returns:
    --------
    distances : np.ndarray
        Array of geodesic distances from the source vertex to all other vertices.
    """
    # Create a TriangleMesh instance for GeomFum
    mesh_geomfum = TriangleMesh(np.array(mesh.vertices), np.array(mesh.faces))

    # Initialize Heat Distance Metric
    solver = geomfum.MeshHeatSolver(mesh_geomfum)

    # Compute distances for each landmark
    distances = []
    for idx in source_index:
        dist = heat.compute_distance(int(idx))
        distances.append(dist)

    return np.array(distances)


def compute_wks(mesh, num_desc=20, hks=False, ldmk=False, indices=None):
    """
    Compute Wave Kernel Signature (WKS) for the mesh.
    Parameters:
    -----------
    mesh : trimesh.Trimesh
        The input mesh.
    num_desc : int
        Number of WKS descriptors to compute.
    hks : bool
        If True, compute Heat Kernel Signature instead of WKS.
    ldmk : bool
        If True, concatenate WKS with landmark based descriptors.
    indices:    np.ndarray
        Array of landmark vertex indices to use if ldmk is True.
    Returns:
    --------
    --------
    np.ndarray
        Array of WKS features for the mesh.
    """
    if len(mesh.faces) > 0:
        mesh1_geomfum = TriangleMesh(np.array(mesh.vertices), np.array(mesh.faces))

        spectrum_finder = LaplacianSpectrumFinder(
            nonzero=False,
            fix_sign=False,
            laplacian_finder=LaplacianFinder.from_registry(which="robust"),
        )
        eigvals, eigvecs = spectrum_finder(
            mesh1_geomfum, as_basis=False, recompute=True
        )

        if hks:
            if ldmk:
                mesh1_geomfum.landmark_indices = indices
                wks = [
                    HeatKernelSignature.from_registry(n_domain=200),
                    LandmarkHeatKernelSignature.from_registry(n_domain=200),
                    ArangeSubsampler(subsample_step=int(400 / num_desc)),
                ]
            else:
                wks = [
                    HeatKernelSignature.from_registry(n_domain=200),
                    ArangeSubsampler(subsample_step=int(200 / num_desc)),
                ]
        else:
            if ldmk:
                wks = [
                    WaveKernelSignature.from_registry(n_domain=200),
                    LandmarkWaveKernelSignature.from_registry(n_domain=200),
                    ArangeSubsampler(subsample_step=int(400 / num_desc)),
                ]
            else:
                wks = [
                    WaveKernelSignature.from_registry(n_domain=200),
                    ArangeSubsampler(subsample_step=int(200 / num_desc)),
                ]

        wks = DescriptorPipeline(wks)
    else:
        mesh1_geomfum = PointCloud(np.array(mesh.vertices))

        spectrum_finder = LaplacianSpectrumFinder(
            nonzero=False,
            fix_sign=False,
            laplacian_finder=LaplacianFinder.from_registry(mesh=False, which="robust"),
        )
        eigvals, eigvecs = spectrum_finder(
            mesh1_geomfum, as_basis=False, recompute=True
        )

        if hks:
            if ldmk:
                mesh1_geomfum.landmark_indices = indices
                wks = [
                    HeatKernelSignature.from_registry(n_domain=200),
                    LandmarkHeatKernelSignature.from_registry(n_domain=200),
                    ArangeSubsampler(subsample_step=int(400 / num_desc)),
                ]
            else:
                wks = [
                    HeatKernelSignature.from_registry(n_domain=200),
                    ArangeSubsampler(subsample_step=int(200 / num_desc)),
                ]
        else:
            if ldmk:
                wks = [
                    WaveKernelSignature.from_registry(n_domain=200),
                    LandmarkWaveKernelSignature.from_registry(n_domain=200),
                    ArangeSubsampler(subsample_step=int(400 / num_desc)),
                ]
            else:
                wks = [
                    WaveKernelSignature.from_registry(n_domain=200),
                    ArangeSubsampler(subsample_step=int(200 / num_desc)),
                ]

        wks = DescriptorPipeline(wks)

    return wks.apply(mesh1_geomfum).T


def compute_biharmonic_distances(mesh, source_index):
    """
    Compute biharmonic distances from a source vertex to all other vertices.

    Parameters:
    -----------
    mesh : trimesh.Trimesh
        The input mesh.
    source_index : int
        Index of the source vertex.
    num_eigenpairs : int
        Number of eigenpairs to compute (controls approximation quality).

    Returns:
    --------
    distances : np.ndarray
        Array of biharmonic distances from source to all vertices.
    """
    if len(mesh.faces) > 0:
        mesh1_geomfum = TriangleMesh(np.array(mesh.vertices), np.array(mesh.faces))

        spectrum_finder = LaplacianSpectrumFinder(
            nonzero=False,
            fix_sign=False,
            laplacian_finder=LaplacianFinder.from_registry(k=50, which="robust"),
        )
        eigvals, eigvecs = spectrum_finder(
            mesh1_geomfum, as_basis=False, recompute=True
        )

    else:
        mesh1_geomfum = PointCloud(np.array(mesh.vertices))

        spectrum_finder = LaplacianSpectrumFinder(
            nonzero=False,
            fix_sign=False,
            laplacian_finder=LaplacianFinder.from_registry(
                k=50, mesh=False, which="robust"
            ),
        )
        eigvals, eigvecs = spectrum_finder(
            mesh1_geomfum, as_basis=False, recompute=True
        )
    # 3) Sort and drop the zero eigenvalue/eigenvector
    evals = eigvals[1:]
    evecs = eigvecs[:, 1:]

    # 4) Compute the Biharmonic distance formula
    #    d_bihar(p, s) = sqrt( Σ_i (φ_i(p) - φ_i(s))² / λ_i² )
    # suppose source_index is a list/array of S indices
    src_idx = np.atleast_1d(source_index)  # shape (S,)
    phis = evecs[src_idx, :]  # shape (S, k)
    # Now compute diffs as (N, S, k):
    diffs = evecs[:, None, :] - phis[None, :, :]
    # shape: (N,1,k) - (1,S,k) → (N,S,k)

    # Then weighted square:
    inv2 = 1.0 / (evals**2)  # shape (k,)
    sq = (diffs**2) * inv2[None, None, :]  # shape (N,S,k)

    # Finally distances: for each source j, distances[:,j]
    distances = np.sqrt(sq.sum(axis=2))  # shape (N,S)

    return distances.T
