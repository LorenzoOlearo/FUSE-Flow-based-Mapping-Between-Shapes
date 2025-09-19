"""Mesh utilities for features computation and sampling"""

import numpy as np
import torch
import trimesh
import os
import networkx as nx

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from sklearn import neighbors
from sklearn.manifold import MDS

# Add this to a new cell in your notebook
import numpy as np
from scipy.sparse.csgraph import shortest_path
from scipy.sparse import coo_matrix
import potpourri3d as pp3d


from geomfum.shape.mesh import TriangleMesh
from geomfum.shape.point_cloud import PointCloud

from geomfum.descriptor.spectral import WaveKernelSignature
from geomfum.descriptor.pipeline import DescriptorPipeline, ArangeSubsampler
from geomfum.laplacian import LaplacianFinder, LaplacianSpectrumFinder


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


def normalize_mesh_08(mesh):
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


def sample_initial_distribution(num_points: int, embedding_dim: int, distribution: str, device: str):
    """
    Generate random samples based on the specified distribution.
    Args:
        args (argparse.Namespace): Command-line arguments containing distribution type and other parameters.
        device (str): Device to use for tensor operations ('cuda' or 'cpu').
    Returns:
        torch.Tensor: Generated samples of shape (num_points, embedding_dim).
    """
    if distribution == "gaussian":
        samples = torch.randn(num_points, embedding_dim).to(device)

    elif distribution == "sphere":
        if embedding_dim == 3:
            samples = sample_sphere_volume(
                radius=1,
                center=(0, 0, 0),
                num_points=num_points,
                device=device,
            )
        else:
            raise NotImplementedError(
                "Sampling from a sphere with dimensions other than 3 is not implemented."
            )
    else:
        raise ValueError(f"Unsupported distribution type: {distribution}")

    return samples


############################### FUNCTIONS FOR MESH EMBEDDINGS #############################
def generate_embeddings(mesh, embedding_type, num_points, features, device: str) -> torch.Tensor:
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
            embedding = get_interpolated_feats(mesh, features, num_points, device=device)

            if embedding_type == "features_only":
                # Drop XYZ coords, keep only feature channels
                embedding = embedding[:, 3:]

        elif embedding_type == "xyz":
            # Sample 3D points directly from mesh surface
            samples, _ = trimesh.sample.sample_surface(mesh, num_points)
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
            feat_tensor = features if torch.is_tensor(features) else torch.tensor(features, dtype=torch.float32, device=device)
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
    samples, face_indices = trimesh.sample.sample_surface(mesh, num_points)
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
        return None
    if args.features_type == "landmarks":
        if len(mesh.faces) > 0:
            features = torch.tensor(
                compute_geodesic_distances(mesh, source_index=np.array(args.landmarks))
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
    elif args.features_type == "wks":
        features = torch.tensor(compute_wks(mesh)).to(device)
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

    # Compute distances for each landmark
    distances = []
    for idx in source_index:
        distances.append([solver.compute_distance(idx)])

    return np.concatenate(distances, axis=0)


def compute_geodesic_distances(mesh, source_index):
    """
    Compute geodesic distances from a source vertex to all other vertices.

    Parameters:
    -----------
    mesh : trimesh.Trimesh
        The mesh on which to compute geodesic distances
    source_index : int
        Index of the source vertex

    Returns:
    --------
    distances : np.ndarray
        Array of distances from source to all vertices
    """
    # Get unique edges
    edges = mesh.edges_unique

    # Calculate edge lengths
    edge_lengths = np.linalg.norm(
        mesh.vertices[edges[:, 0]] - mesh.vertices[edges[:, 1]], axis=1
    )

    # Create a sparse adjacency matrix
    n_vertices = len(mesh.vertices)
    graph = coo_matrix(
        (edge_lengths, (edges[:, 0], edges[:, 1])), shape=(n_vertices, n_vertices)
    )

    # Make the graph symmetric (undirected)
    graph = graph + graph.T

    # Compute shortest paths
    distances, predecessors = shortest_path(
        csgraph=graph, directed=False, indices=source_index, return_predecessors=True
    )

    return distances


def compute_wks(mesh):
    """
    Compute Wave Kernel Signature (WKS) for the mesh.
    Parameters:
    -----------
    mesh : trimesh.Trimesh
        The input mesh.
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

        wks = [
            WaveKernelSignature.from_registry(n_domain=20),
            ArangeSubsampler(subsample_step=4),
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

        wks = [
            WaveKernelSignature.from_registry(n_domain=20),
            ArangeSubsampler(subsample_step=4),
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
