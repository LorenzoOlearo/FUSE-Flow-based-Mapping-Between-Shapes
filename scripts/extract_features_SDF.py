import os
from pathlib import Path
import argparse
import re

from typing import List, Tuple, Optional

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
            zaxis_title='Z',
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    if save_path is not None:
        fig.write_html(f"{save_path}.html")
        fig.write_image(f"{save_path}.png", width=800, height=600)
    else:
        fig.show()


def normalize_mesh(mesh):
    centroid = torch.tensor(mesh.centroid, dtype=torch.float32)
    verts = torch.tensor(mesh.vertices, dtype=torch.float32)

    verts -= centroid
    verts /= verts.abs().max()
    verts *= 0.8

    mesh.vertices = verts.numpy()
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
        scene=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False, aspectmode='data'),
        scene2=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False, aspectmode='data'),
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
    print(f'query coord max: {query_coord.max(axis=0)}, min: {query_coord.min(axis=0)}')
    print(f'occupied voxel max: {occupied_voxels.max(axis=0)}, min: {occupied_voxels.min(axis=0)}')
    
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


def extract_surface_mask(sdf_path: Path, eps: float, device: str):
    """ Extracts the surface mask from the SDF model. """
    mlp_cfg = MLPConfig()
    sdf_model = MLP(mlp_cfg).to(device)
    sdf_model.load_state_dict(torch.load(sdf_path))
    print(f"Loaded SDF model from {sdf_path}")
    
    sdf_volume = evaluate_model(sdf_model, make_volume(device))
    surface_mask = np.argwhere(np.abs(sdf_volume) <= eps)
    surface_mask = surface_mask.T.cpu().numpy()



def get_targets(args) -> List[str]:
    """Determine which targets to process based on CLI arguments."""
    sdf_dir = Path('./out/SDFs')
    if args.all:
        targets = [
            f.name for f in sdf_dir.iterdir() if f.is_dir() and
            any(child.suffix == '.pth' for child in f.iterdir()) and
            any(child.name.endswith('-landmarks-voxels.npy') for child in f.iterdir())
        ]
        print(f"Processing all targets: {targets}")
    elif args.test:
        targets = [
            f.name for f in sdf_dir.iterdir() if f.is_dir() and
            any(child.suffix == '.pth' for child in f.iterdir()) and
            any(child.name.endswith('-landmarks-voxels.npy') for child in f.iterdir()) and
            (not args.test or any(80 <= int(num) <= 99 for num in re.findall(r'\d+', f.name)))
        ]
    elif args.target:
        targets = [args.target]
    else:
        raise ValueError("Specify --all or provide a --target.")
    return targets


def load_target_data(target: str, eps: float, device: str) -> Tuple[Path, Path, np.ndarray]:
    """For a given target, load the landmarks, the neural SDF model and extract its voxel surface mask."""
    sdf_path = Path(f'./out/SDFs/{target}/{target}-SDF.pth')
    landmarks_path = Path(f'./out/SDFs/{target}/{target}-landmarks-voxels.npy')
    print(f"Target: {target}\nSDF path: {sdf_path}\nLandmarks path: {landmarks_path}")
    landmarks = np.load(landmarks_path)
    print(f"Loaded landmarks: {landmarks.shape}\n{landmarks}")

    mlp_cfg = MLPConfig()
    sdf_model = MLP(mlp_cfg).to(device)
    sdf_model.load_state_dict(torch.load(sdf_path))
    print(f"Loaded SDF model from {sdf_path}. Evaluating volume...")
    sdf_volume = evaluate_model(sdf_model, make_volume(device))
    surface_mask = np.argwhere(np.abs(sdf_volume) <= eps)
    surface_mask = surface_mask.T.cpu().numpy()

    return landmarks, surface_mask, sdf_model


def compute_voxel_geodesics(surface_mask, landmarks, h, eps, target: str, additional_plots=False) -> List[np.ndarray]:
    """Compute geodesic distances for each landmark voxel."""
    dists = []

    for i, src_vox in enumerate(landmarks):
        print(f"Landmark {i}: {src_vox}")
        geo_dist = geodesic_dijkastra(surface_mask, src_vox, h=h)
        if geo_dist.max() == 0:
            print(f"Warning: Landmark {i} has no reachable voxels, exiting.")
            exit(1)
        else:
            dists.append(geo_dist)

        if additional_plots is True:
            plot_points(
                surface_mask,
                distances=geo_dist,
                title=f"Geodesic distances from landmark {i}",
                save_path=f"out/SDFs/{target}/{target}-geodesic-dijkstra-landmark-{i}",
            )

    max_dist = max(np.max(dist) for dist in dists)
    dists = [dist / max_dist for dist in dists]
    for i, dist in enumerate(dists):
        print(f"Landmark {i} distances: min={np.min(dist)}, max={np.max(dist)}, mean={np.mean(dist)}")

    return dists


def points_dist_in_grid(surface_points, surface_mask, dists, target: str, resolution, min_coord, max_coord, additional_plots=False):
    """Given a set of surface points, find the closest occupied voxel indices and sample distances."""
    surface_voxs = find_closest_occupied_index(
        surface_mask,
        surface_points,
        min_coord,
        max_coord,
        resolution,
        world_to_grid=True,
        unique=False
    )

    voxel_to_dist_map = {
        tuple(coord): [dists[l][idx] for l in range(len(dists))]
        for idx, coord in enumerate(surface_mask)
    }

    dists_surface_points = np.array([voxel_to_dist_map[tuple(vox)] for vox in surface_voxs])

    if additional_plots is True:
        plot_points(surface_mask, distances=dists[0], title="Dijkstra Geodesic Distances on surface voxels", save_path=f"out/SDFs/{target}/{target}-dijkstra-on-voxels")

    return dists_surface_points


def extract_mesh_dists(args, surface_mask, dists, target, resolution, min_coord, max_coord):
    if (args.all is True or args.test is True) and args.mesh_folder is not None and args.extension is not None:
        mesh_path = os.path.join(args.mesh_folder, target + f".{args.extension}")
        print(f"Loading mesh from {mesh_path}")
        mesh = trimesh.load(mesh_path, force='mesh', process=False)
        print(f"Mesh loaded with {len(mesh.vertices)} vertices and {len(mesh.faces)} faces.")
        mesh = normalize_mesh(mesh)
        dists_mesh = points_dist_in_grid(mesh.vertices, surface_mask, dists, target, resolution, min_coord, max_coord, args.additional_plots)
        if args.additional_plots:
            plot_points(mesh.vertices, distances=dists_mesh[:, 0], title="Dijkstra Distances on Mesh Vertices", save_path=f"out/SDFs/{target}/{target}-dijkstra-on-mesh-vertices")
        print(f"Mesh distances shape: {dists_mesh.shape}")
    elif (args.all is False or args.test is False) and args.mesh_path is not None:
        mesh_path = Path(args.mesh_path)
        print(f"Loading mesh from {mesh_path}")
        mesh = trimesh.load(mesh_path, force='mesh', process=False)
        print(f"Mesh loaded with {len(mesh.vertices)} vertices and {len(mesh.faces)} faces.")
        mesh = normalize_mesh(mesh)
        dists_mesh = points_dist_in_grid(mesh.vertices, surface_mask, dists, target, resolution, min_coord, max_coord, args.additional_plots)
        if args.additional_plots:
            plot_points(mesh.vertices, distances=dists_mesh[:, 0], title="Dijkstra Distances on Mesh Vertices", save_path=f"out/SDFs/{target}/{target}-dijkstra-on-mesh-vertices")
        print(f"Mesh distances shape: {dists_mesh.shape}")
    else:
        print("No mesh provided, or --all is False and no mesh_path specified. Skipping mesh distance computation.")
        print(f"args.all: {args.all}, args.mesh_path: {args.mesh_path}, args.mesh_folder: {args.mesh_folder}, args.extension: {args.extension}")
        exit(1)

    return dists_mesh, mesh


def project_to_surface(points, sdf_model, device):
    sdf_model_cfg = NeuralSDFConfig()
    neural_SDF = NeuralSDF(config=sdf_model_cfg, network=sdf_model)

    points = torch.tensor(points, dtype=torch.float32, device=device)
    projected = neural_SDF.project_nearest(points)
    projected = projected.detach().cpu().numpy()

    return projected


def project_landmarks_to_surface(landmark_indices, mesh, sdf_model, device):
    """
    Projects landmark indices to the surface of the SDF model.
    """
    landmarks = mesh.vertices[landmark_indices]
    projected_landmarks = project_to_surface(landmarks, sdf_model, device)

    return projected_landmarks


def main(args):
    device = "cuda:0"
    resolution = 512
    min_coord = -1.0
    max_coord = 1.0
    h = (max_coord - min_coord) / (resolution - 1)
    eps = h  # Equal to voxel spacing
    landmark_indices = [412, 5891, 6593, 3323, 2119]  # TODO: Configurable

    targets = get_targets(args)
    print(f"Processing {len(targets)} targets: {targets}")

    for target in targets:
        landmarks, surface_mask, sdf_model = load_target_data(target, eps, device)

        dists = compute_voxel_geodesics(surface_mask, landmarks, h, eps, target, args.additional_plots)

        sdf_model_cfg = NeuralSDFConfig()
        neural_SDF = NeuralSDF(config=sdf_model_cfg, network=sdf_model)

        print(f"Sampling zero level set with eps={eps}, h={h}, resolution={resolution}")
        surface_points = neural_SDF.sample_zero_level_set(
            num_samples=args.num_points,
            threshold=eps,
            samples_per_step=100000
        ).detach().cpu().numpy()
        print(f"Sampled {len(surface_points)} points from SDF - shape: {surface_points.shape}")

        dists_surface_points = points_dist_in_grid(surface_points, surface_mask, dists, target, resolution, min_coord, max_coord, args.additional_plots)
        if args.additional_plots:
            if args.num_points > 100000:
                print("Sampling 100k points for plotting...")
                sample_indices = np.random.choice(len(surface_points), size=100000, replace=False)
                surface_points_plot = surface_points[sample_indices]
                plot_points(surface_points_plot, distances=dists_surface_points[sample_indices, 0], title="Dijkstra Distances on SDF-sampled points", save_path=f"out/SDFs/{target}/{target}-sdf-dijkstra-sampled-points")
            else:
                print("Plotting all sampled points...")
                plot_points(surface_points, distances=dists_surface_points[:, 0], title="Dijkstra Distances on SDF-sampled points", save_path=f"out/SDFs/{target}/{target}-sdf-dijkstra-sampled-points")
        print(f"Features shape: {dists_surface_points.shape}")

        mesh_dists, mesh = extract_mesh_dists(args, surface_mask, dists, target, resolution, min_coord, max_coord)

        print(f"Number of surface voxels: {len(surface_mask)}")
        print(f"Surface points shape: {surface_points.shape} - number of surface points: {len(surface_points)}")
        print(f"Dijkstra distances shape: {dists_surface_points.shape} - number of distances: {len(dists_surface_points)}")

        print(f"Projecting mesh vertices to SDF zero level set...")
        verts_projection = project_to_surface(mesh.vertices, sdf_model, device)
        np.savetxt(f"out/SDFs/{target}/{target}-mesh-vertices-projected.txt", verts_projection, fmt='%.6f')
        mesh_dists_projected = points_dist_in_grid(verts_projection, surface_mask, dists, target, resolution, min_coord, max_coord, args.additional_plots)

        proj_diff = np.abs(mesh_dists - mesh_dists_projected)
        print(f"Projection difference: min={np.min(proj_diff)}, max={np.max(proj_diff)}, mean={np.mean(proj_diff)}")

        out_dir = Path(f"out/SDFs/{target}")
        out_dir.mkdir(exist_ok=True)
        np.savetxt(out_dir / f"{target}-sdf-dijkstra-features.txt", dists_surface_points, fmt='%.6f')
        np.savetxt(out_dir / f"{target}-sdf-sampled-points.txt", surface_points, fmt='%.6f')
        np.savetxt(out_dir / f"{target}-sdf-mesh-dists.txt", mesh_dists, fmt='%.6f')
        np.savetxt(out_dir / f"{target}-sdf-mesh-dists-projected.txt", mesh_dists_projected, fmt='%.6f')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate SDF model on mesh")
    parser.add_argument('--target', type=str, help="Target of the feature extraction, should be a filename without extension and will be searched in the out/SDFs directory")
    parser.add_argument('--additional_plots', action='store_true', default=False, help="Plot the volume of sampled points, plot the volume of voxelized points, plot the geodesic distances between these points and the landmarks")
    parser.add_argument('--num_points', type=int, default=100000, help="Number of points to sample on the zero level set of the SDF")
    parser.add_argument('--all', action='store_true', help="Sample and extract features for all [off/ply] file in the target directory")
    parser.add_argument('--test', action='store_true', help="Sample and extract features only from the last 20 targets (80-99) in the out/SDFs directory")
    parser.add_argument('--mesh_path', type=str, default=None, help="Path to the target mesh file to be processed")
    parser.add_argument('--mesh_folder', type=str, default=None, help="Folder containing the target mesh files to be processed. If not provided, will use the default out/SDFs directory.")
    parser.add_argument('--extension', type=str, default='ply', help="File extension of the target mesh files (default: 'ply')")

    args = parser.parse_args()

    print("SDF Feature Extraction:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print("-----------------------------------------------")

    main(args)
