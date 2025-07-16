import os
from pathlib import Path
import argparse

import numpy as np
import torch
import trimesh

import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import shortest_path

from train_SDF import MLP, MLPConfig, make_volume, evaluate_model, NeuralSDF, NeuralSDFConfig


def load_sdf_volume(path: Path, device: str):
    mlp_cfg = MLPConfig()
    sdf_model = MLP(mlp_cfg).to(device)

    sdf_model.load_state_dict(torch.load(path))
    sdf_volume = evaluate_model(sdf_model, make_volume(device))
    return sdf_volume, sdf_model


def plot_points(points, distances=None, title="3D Points", save_path=None, colorbar_title="Distance"):
    """
    Plot 3D points, optionally colored by a distance array.

    Args:
        points (np.ndarray): (N, 3) array of point coordinates.
        distances (np.ndarray or None): (N,) array of values to color points. If None, use blue.
        title (str): Plot title.
        save_path (str or None): If given, save HTML and PNG.
        colorbar_title (str): Title for the colorbar if distances is provided.
    """
    if distances is not None:
        marker_dict = dict(
            size=3,
            color=distances,
            colorscale='Viridis',
            colorbar=dict(title=colorbar_title),
            opacity=0.8
        )
    else:
        marker_dict = dict(size=2, color='blue', opacity=0.8)

    fig = go.Figure(data=[go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers',
        marker=marker_dict
    )])

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    if save_path is not None:
        fig.write_html(f"{save_path}.html")
        fig.write_image(f"{save_path}.png", width=800, height=600)
    else:
        fig.show()


def normalize_mesh(mesh):
    rescale = max(mesh.extents) / 2.
    tform = [
        -(mesh.bounds[1][i] + mesh.bounds[0][i]) / 2.
        for i in range(3)
    ]
    matrix = np.eye(4)
    matrix[:3, 3] = tform
    mesh.apply_transform(matrix)
    matrix = np.eye(4)
    matrix[:3, :3] /= rescale
    mesh.apply_transform(matrix)
    
    return mesh


def compute_mesh_geodesic_distances(mesh, source_index):
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


def plot_slices(X, title="Heat Distribution Slices", save_path="slices.png"):
    # Center slice for each axis
    level_xy = X.shape[0] // 2
    level_xz = X.shape[1] // 2
    level_yz = X.shape[2] // 2

    slice_xy = X[level_xy, :, :]
    slice_xz = X[:, level_xz, :]
    slice_yz = X[:, :, level_yz]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    im0 = axes[0].imshow(slice_xy, cmap='RdBu')
    axes[0].set_title("Heat Slice at center z (XY plane)")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(slice_xz, cmap='RdBu')
    axes[1].set_title("Heat Slice at center y (XZ plane)")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(slice_yz, cmap='RdBu')
    axes[2].set_title("Heat Slice at center x (YZ plane)")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def plot_geodesic_comparison(vertices, faces, true_dists, dijkstra_dists, save_path):
    """
    Plot a mesh with two subplots: true geodesics vs Dijkstra-based distances.

    Parameters:
        vertices (Nx3): Vertex coordinates
        faces (Mx3): Triangle indices
        true_dists (N,): Ground truth geodesic distances
        dijkstra_dists (N,): Approximated distances from voxel-based Dijkstra
        save_path (str): Path to save the HTML file
    """
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    i, j, k = faces[:, 0], faces[:, 1], faces[:, 2]

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'surface'}, {'type': 'surface'}]],
        subplot_titles=("True Geodesic Distances", "Dijkstra Approximation")
    )

    mesh_true = go.Mesh3d(
        x=x, y=y, z=z,
        i=i, j=j, k=k,
        intensity=true_dists,
        colorscale="Viridis",
        intensitymode="vertex",
        colorbar=dict(title="Distance", x=0.45),
        name="True",
        showscale=True,
        lighting=dict(ambient=0.5, diffuse=0.9)
    )

    mesh_dijkstra = go.Mesh3d(
        x=x, y=y, z=z,
        i=i, j=j, k=k,
        intensity=dijkstra_dists,
        colorscale="Viridis",
        intensitymode="vertex",
        colorbar=dict(title="Distance", x=1.0),
        name="Dijkstra",
        showscale=True,
        lighting=dict(ambient=0.5, diffuse=0.9)
    )

    fig.add_trace(mesh_true, row=1, col=1)
    fig.add_trace(mesh_dijkstra, row=1, col=2)

    fig.update_layout(
        title_text="Geodesic Distance Comparison",
        scene=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False),
        scene2=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False),
        margin=dict(l=0, r=0, t=40, b=0)
    )

    fig.write_html(f"{save_path}.html")
    fig.write_image(f"{save_path}.png", width=1600, height=800)


def convert_world_to_grid(coords, min_coord, max_coord, resolution, unique=False):
    """
    Maps a world-space point to voxel index space.
    """
    coords = np.asarray(coords, dtype=np.float32)
    min_coord = np.asarray(min_coord)
    max_coord = np.asarray(max_coord)

    normalized = (coords - min_coord) / (max_coord - min_coord)
    index = np.clip(normalized * (resolution - 1), 0, resolution - 1)
    voxel_indices = np.round(index).astype(np.int32)

    if unique:
        unique_voxel_indices = np.unique(voxel_indices, axis=0)
        print(f"Unique voxel indices: {unique_voxel_indices.shape[0]} out of {voxel_indices.shape[0]} total")
        return unique_voxel_indices
    else:
        return voxel_indices


def find_closest_occupied_index(occupied_voxels, query_coord, min_coord, max_coord, resolution, world_to_grid=False, unique=False):
    """
    Find the index or indices of the voxel(s) in `occupied_voxels` that are closest to `query_coord`,
    after mapping it/them into voxel index space.

    If query_coord is an array of points, returns an array of indices.
    """
    if world_to_grid:
        query_coord = convert_world_to_grid(query_coord, min_coord, max_coord, resolution, unique=unique)
    occupied_voxels = np.asarray(occupied_voxels)
    query_coord = np.asarray(query_coord)
    
    # Single point
    if query_coord.ndim == 1:
        diffs = occupied_voxels - query_coord
        dists = np.linalg.norm(diffs, axis=1)
        closest_idx = np.argmin(dists)
        return closest_idx
    
    # Multiple points: for each query, find closest occupied voxel
    else:
        tree = cKDTree(occupied_voxels)
        _, idxs = tree.query(query_coord)
        return occupied_voxels[idxs]


def geodesic(sdf, src_pt, eps, h, min_coord, max_coord, resolution):
    occupancy = np.array(sdf.abs() <= eps).astype(np.uint8)
    occupied_voxels = np.argwhere(occupancy == 1)
    index_grid = -np.ones_like(occupancy, dtype=int)
    for idx, (x, y, z) in enumerate(occupied_voxels):
        index_grid[x, y, z] = idx
        
    # Consider all 26 neighbors (including diagonals)
    neighbor_offsets = np.array([
        [dx, dy, dz]
        for dx in [-1, 0, 1]
        for dy in [-1, 0, 1]
        for dz in [-1, 0, 1]
        if not (dx == 0 and dy == 0 and dz == 0)
    ])

    rows, cols = [], []

    shape = occupancy.shape

    # Computes the neighbors for each occupied voxel
    for offset in neighbor_offsets:
        neighbors = occupied_voxels + offset

        # Remove neighbors that are out of bounds
        valid_mask = np.all((neighbors >= 0) & (neighbors < shape), axis=1)

        src_vox = occupied_voxels[valid_mask]
        dst_vox = neighbors[valid_mask]

        src_idx = index_grid[src_vox[:, 0], src_vox[:, 1], src_vox[:, 2]]
        dst_idx = index_grid[dst_vox[:, 0], dst_vox[:, 1], dst_vox[:, 2]]

        # Only keep edges between occupied voxels (index != -1)
        mask = dst_idx != -1
        rows.extend(src_idx[mask])
        cols.extend(dst_idx[mask])

    # Weight edges by Euclidean distance (1 for 6-connectivity)
    weights = np.ones(len(rows))

    # Construct sparse adjacency matrix
    N = len(occupied_voxels)
    adj = coo_matrix((weights, (rows, cols)), shape=(N, N)).tocsr()

    # Compute geodesic distance matrix (all-pairs) using Dijkstra's algorithm
    src_idx = find_closest_occupied_index(occupied_voxels, src_pt, min_coord, max_coord, resolution)

    print(f"Source point: {src_pt}")
    print(f"Source point index: {src_idx}, coordinates: {occupied_voxels[src_idx]}")
    if src_idx == -1:
        raise ValueError("Source point is not in the occupied voxel grid.")
    dist_from_source = shortest_path(
        csgraph=adj, indices=src_idx, directed=False, method="D", unweighted=False
    )

    dist_grid = np.zeros_like(occupancy, dtype=np.float32)
    for i, (x, y, z) in enumerate(occupied_voxels):
        dist = dist_from_source[i]
        if np.isfinite(dist):
            dist_grid[x, y, z] = h * dist

    return dist_grid


def geodesic_dijkastra(surface_voxs, landmark_vox, h):
    coord_to_index = {tuple(coord): idx for idx, coord in enumerate(surface_voxs)}
    neighbor_offsets = np.array([
        [dx, dy, dz]
        for dx in [-1, 0, 1]
        for dy in [-1, 0, 1]
        for dz in [-1, 0, 1]
        if (dx, dy, dz) != (0, 0, 0)
    ])
    weights = np.linalg.norm(neighbor_offsets, axis=1)

    rows, cols, all_weights = [], [], []
    for idx, voxel in enumerate(surface_voxs):
        if idx % 1000 == 0:
            print(f"> Processing voxel {idx}/{len(surface_voxs)}")
        for offset, weight in zip(neighbor_offsets, weights):
            neighbor_coord = tuple(voxel + offset)
            if neighbor_coord in coord_to_index:
                dst_idx = coord_to_index[neighbor_coord]
                rows.append(idx)
                cols.append(dst_idx)
                all_weights.append(weight)

    N = len(surface_voxs)
    adj = coo_matrix((all_weights, (rows, cols)), shape=(N, N)).tocsr()

    # Find closest surface voxel to landmark_vox
    tree = cKDTree(surface_voxs)
    _, landmark_closest_idx = tree.query(landmark_vox)
    print(f"Landmark voxel: {landmark_vox}, closest index: {landmark_closest_idx} at coordinates {surface_voxs[landmark_closest_idx]}")

    dist_from_source = shortest_path(
        csgraph=adj, indices=landmark_closest_idx, directed=False, method="D", unweighted=False
    )

    return dist_from_source * h


def main(args):
    device = "cuda:1"
    mesh_comparison = True
    min_coord: float = -1.0
    max_coord: float =  1.0
    resolution: int = 512

    target = args.target

    sdf_path: Path = Path(f'./out/SDFs/{target}-SDF.pth')
    landmarks_path = Path(f'./out/SDFs/{target}-landmarks-voxels.npy')

    # TODO: make this configurable
    if mesh_comparison:
        landmarks_indices = [412, 5891,6593,3323,2119]
        mesh_filename: str = 'tr_reg_009.ply'
        mesh_path: Path = Path(f'./data/MPI-FAUST/training/registrations/{target}.ply')
        mesh = trimesh.load(mesh_path, force='mesh')

    h = (max_coord - min_coord) / (resolution - 1)
    eps = 1 * h

    mlp_cfg = MLPConfig()
    sdf_model = MLP(mlp_cfg).to(device)
    sdf_model.load_state_dict(torch.load(sdf_path))
    print(f"Loaded SDF model from {sdf_path}")
    
    sdf_volume = evaluate_model(sdf_model, make_volume(device))
    surface_mask = np.argwhere(np.abs(sdf_volume) <= eps)
    surface_mask = surface_mask.T.cpu().numpy()
    
    landmarks = np.load(landmarks_path)
    print(f"Loaded landmarks: {landmarks.shape}, at voxel coordinates: \n{landmarks}")

    dists = []
    for i in range(len(landmarks)):
        src_vox = landmarks[i]
        print(f"Landmark {i}: {src_vox}")
        geo_dists_dijkastra = geodesic_dijkastra(
            surface_mask,
            src_vox,
            h=h,
        ) 
        geo_dists_dijkastra = geo_dists_dijkastra / geo_dists_dijkastra.max()
        dists.append(geo_dists_dijkastra)

        if args.additional_plots:
            plot_points(
                surface_mask,
                distances=geo_dists_dijkastra,
                title=f"Geodesic distances from landmark {i}",
                save_path=f"out/{target}-geodesic-dijkstra-landmark-{i}"
            )
       
        save_path = f"out/{target.split}"
        if mesh_comparison:
            geo_dists = compute_mesh_geodesic_distances(mesh, landmarks_indices[i])
            geo_dists /= geo_dists.max()
            plot_geodesic_comparison(mesh.vertices, mesh.faces, geo_dists, geo_dists_dijkastra, save_path=f"{save_path}-geodesic-comparison-landmark-{i}")
    
    neural_sdf_cfg = NeuralSDFConfig()
    neural_sdf = NeuralSDF(config=neural_sdf_cfg, network=sdf_model)
    print(f"Sampling zero level set with eps={eps}, h={h}, resolution={resolution}")
    surface_points = (neural_sdf.sample_zero_level_set(num_samples=args.num_points, threshold=eps, samples_per_step=100000)).detach().cpu().numpy()
    surface_voxs = find_closest_occupied_index(surface_mask, surface_points, min_coord, max_coord, resolution, world_to_grid=True, unique=False)
    
    voxel_to_dist_map = {tuple(coord): [dists[landmark_idx][idx] for landmark_idx in range(len(dists))] for idx, coord in enumerate(surface_mask)}
    
    dists_sampled = []
    for i, voxel in enumerate(surface_voxs):
        dists_sampled.append(voxel_to_dist_map[tuple(voxel)])
    dists_sampled = np.array(dists_sampled)

    if args.additional_plots:
        plot_points(
            surface_mask,
            distances=dists[0],
            title="Dijkstra Geodesic Distances on surface voxels",
            save_path=f"out/{target}-dijkstra-on-voxels"
        )
        
        plot_points(
            surface_points,
            distances=dists_sampled[:, 0],
            title="Dijkstra Geodesic Distances on points sampled from SDF isosurface",
            save_path=f"out/{target}-sdf-dijkstra-sampled-points"
        )
        
    os.makedirs("out/features", exist_ok=True)
    np.savetxt(f"out/features/{target}-sdf-dijkstra-features.txt", dists_sampled, fmt='%.6f')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate SDF model on mesh")
    parser.add_argument('--mesh_comparison', action='store_true', default=False, help="Compare the geodesic distances computed on the SDF via Dijkstra with the ones computed on the mesh")
    parser.add_argument('--target', type=str, help="Target of the feature extraction, should be a filename without extension and will be searched in the out/SDFs directory")
    parser.add_argument('--additional_plots', action='store_true', default=False, help="Plot the volume of sampled points, plot the volume of voxelized points, plot the geodesic distances between these points and the landmarks")
    parser.add_argument('--num_points', type=int, default=100000, help="Number of points to sample on the zero level set of the SDF")

    args = parser.parse_args()

    main(args)
