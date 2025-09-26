import os
from pathlib import Path
from typing import List
import argparse
import csv
import numpy as np
import torch
import trimesh
import matplotlib.colors as mcolors
import time
from typing import Callable, Dict, Optional
from dataclasses import dataclass
import pandas as pd
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from model.models import FMCond
from model.networks import MLP
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from geomfum.shape.mesh import TriangleMesh
from geomfum.metric.mesh import HeatDistanceMetric
import potpourri3d as pp3d

from util.matching_utils import *
from util.metrics import (
    compute_dirichlet_energy,
    compute_coverage,
    compute_geodesic_error,
)
from util.plot import plot_points, start_end_subplot
from util.mesh_utils import compute_geodesic_distances, generate_embeddings

@dataclass
class DataPath:
    output_dir: Path
    landmarks: List[int]
    dataset_path: Path
    dists_path: Path
    features_path: Path
    dataset_extension: str
    flows_path: Path
    flows_SDFs_path: Path
    sdf_path: Path
    corr_path: Optional[Path] = None


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
        rows=1,
        cols=2,
        specs=[[{"type": "surface"}, {"type": "surface"}]],
        subplot_titles=("True Geodesic Distances", "Dijkstra Approximation"),
    )

    mesh_true = go.Mesh3d(
        x=x,
        y=y,
        z=z,
        i=i,
        j=j,
        k=k,
        intensity=true_dists,
        colorscale="Viridis",
        intensitymode="vertex",
        colorbar=dict(title="Distance", x=0.45),
        name="True",
        showscale=True,
        lighting=dict(ambient=0.5, diffuse=0.9),
    )

    mesh_dijkstra = go.Mesh3d(
        x=x,
        y=y,
        z=z,
        i=i,
        j=j,
        k=k,
        intensity=dijkstra_dists,
        colorscale="Viridis",
        intensitymode="vertex",
        colorbar=dict(title="Distance", x=1.0),
        name="Dijkstra",
        showscale=True,
        lighting=dict(ambient=0.5, diffuse=0.9),
    )

    fig.add_trace(mesh_true, row=1, col=1)
    fig.add_trace(mesh_dijkstra, row=1, col=2)

    fig.update_layout(
        title_text="Geodesic Distance Comparison",
        scene=dict(
            xaxis_visible=False,
            yaxis_visible=False,
            zaxis_visible=False,
            aspectmode="data",
        ),
        scene2=dict(
            xaxis_visible=False,
            yaxis_visible=False,
            zaxis_visible=False,
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
    )

    fig.write_html(f"{save_path}.html")
    fig.write_image(f"{save_path}.png", width=1600, height=800)


def create_rgb_colormap(points):
    """
    Create a colormap for the points based on their coordinates.
    The color is determined by the distance from the origin.
    """
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    points = (points - points.min(axis=0)) / (points.max(axis=0) - points.min(axis=0))
    colors_hex = [mcolors.rgb2hex(c) for c in points]

    return colors_hex


def normalize_mesh(mesh):
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


def is_in_range(name):
    try:
        num = int("".join(filter(str.isdigit, name)))
        return 80 <= num <= 100
    except ValueError:
        return False


def get_targets_faust(args) -> List[str]:
    """Determine which targets to process."""
    # targets = [
    #     f.name for f in SDFs_PATH.iterdir() if f.is_dir() and
    #     any(child.name.endswith('-features.txt') for child in f.iterdir()) and
    #     (FLOWS_PATH / f.name / 'checkpoint-9999.pth').is_file() and
    #     is_in_range(f.name)
    # ]

    targets = []
    for i in range(80, 100):
        if args.source_rep == "sdf" or args.target_rep == "sdf":
            if i == 86 or i == 87 or i == 96 or i == 97:
                tqdm.write(f"Skipping topologically incorrect shape tr_reg_{i:03d}")
                continue
        target = f"tr_reg_{i:03d}"
        targets.append(target)

    tqdm.write(f"Processing all targets: {targets}")
    return targets


def get_targets_smal(flows_path) -> List[str]:
    """Determine which targets to process."""
    targets = [
        f.name
        for f in flows_path.iterdir()
        if f.is_dir()
        and f.name.startswith(("cougar", "hippo", "horse"))
    ]

    tqdm.write(f"Processing targets: {targets}")
    return targets


def get_mesh_element_features(
    element: str,
    mesh: trimesh.Trimesh,
    device: str,
    data_path: DataPath,
    recompute: bool
) -> torch.Tensor:

    vertex_features_path = Path(data_path.features_path, element, f"vertex-geodesics.txt")

    if vertex_features_path.exists() and not recompute:
        tqdm.write(f"Loading precomputed features for {element} from {data_path.features_path}")
        vertex_features = torch.tensor(np.loadtxt(vertex_features_path).astype(np.float32)).to(device)
    else:
        tqdm.write(f"Computing features for {element} in {data_path.features_path}")
        vertex_features = torch.tensor(compute_geodesic_distances(mesh, data_path.landmarks).T.astype(np.float32)).to(device)
        vertex_features = vertex_features / vertex_features.max()

        os.makedirs(vertex_features_path.parent, exist_ok=True)
        np.savetxt(vertex_features_path, vertex_features.cpu().numpy())

    return vertex_features


def process_element(
    element: str,
    representation: str,
    device: str,
    mesh_baseline: bool,
    features_normalization: str,
    data_path: DataPath,
):
    # The mesh is needed both for 'mesh' and 'sdf' representations because we
    # need to project the mesh vertices onto the SDF isosurface for evaluation
    mesh_path = Path(data_path.dataset_path, element + data_path.dataset_extension)
    mesh = trimesh.load(mesh_path, process=False)
    points = torch.tensor(mesh.vertices.astype(np.float32)).to(device)
    model = FMCond(
        channels=len(data_path.landmarks),
        network=MLP(channels=len(data_path.landmarks)).to(device),
    )

    if representation == "mesh":
        tqdm.write(f"Loading {element} with {representation} representation")
        vertex_features = get_mesh_element_features(element, mesh, device, data_path, recompute=True)
        model.load_state_dict(
            torch.load(
                Path(data_path.flows_path, element, "checkpoint-9999.pth"),
                weights_only=False,
            )["model"],
            strict=True,
        )

    elif representation == "sdf":
        tqdm.write(f"Loading {element} with {representation} representation")
        features_path = Path(
            data_path.flows_SDFs_path,
            element,
            f"vertex-geodesics-interpolated.txt",
        )
        vertex_features_path = Path(
            data_path.flows_SDFs_path, element, f"vertex-geodesics.txt"
        )
        # if mesh_baseline is False:
        #     features_path = Path(SDFs_PATH, element, f'{element}-sdf-dijkstra-features.txt')
        #     points_path = Path(SDFs_PATH, element, f'{element}-sdf-sampled-points.txt')
        #     points = torch.tensor(np.loadtxt(points_path).astype(np.float32)).to(device)
        # else:
        #     tqdm.write(f"Using mesh baseline for {element}")
        #     features_path = Path(SDFs_PATH, element, f'{element}-sdf-mesh-dists-projected-interpolated.txt')
        #     # features_path = Path(SDFs_PATH, element, f'{element}-sdf-extracted-mesh-dists.txt')   # Analitic SDF features
        #     points = torch.tensor(mesh.vertices.astype(np.float32)).to(device)
        features = torch.tensor(np.loadtxt(features_path).astype(np.float32)).to(device)
        vertex_features = torch.tensor(
            np.loadtxt(vertex_features_path).astype(np.float32)
        ).to(device)
        model.load_state_dict(
            torch.load(
                Path(data_path.flows_SDFs_path, element, "checkpoint-9999.pth"),
                weights_only=False,
            )["model"],
            strict=True,
        )

    else:
        raise ValueError(f"Invalid representation: {representation}")

    model.to(device)
    model.eval()

    return vertex_features, points, model, mesh


@dataclass
class MatchingResult:
    """Holds the output of a P2P method."""

    indices: torch.Tensor
    matched_points: torch.Tensor
    euclidean_error: float
    geodesic_error: float
    dirichlet_energy: float
    coverage: float
    elapsed: float


def get_matching_methods(
    source_features,
    target_features,
    source_path,
    target_path,
    source_model,
    target_model,
    source_landmarks,
    target_landmarks,
    device: str,
    matching_methods: str,
) -> Dict[str, Callable[[], torch.Tensor]]:
    """Return mapping of strategy names to their compute functions."""

    if matching_methods == "fast":
        return {
            "knn": lambda: compute_p2p_with_knn(source_features, target_features),
            "flow": lambda: compute_p2p_with_flows_composition(
                source_features, target_features, source_model, target_model, device
            ),
        }
    elif matching_methods == "all":
        return {
            "knn": lambda: compute_p2p_with_knn(source_features, target_features),
            "ot": lambda: compute_p2p_with_ot(source_features, target_features),
            "flow": lambda: compute_p2p_with_flows_composition(
                source_features, target_features, source_model, target_model, device
            ),
            "flow-zoomout": lambda: compute_p2p_with_flows_composition_zoomout(
                source_path,
                target_path,
                source_features,
                target_features,
                source_model,
                target_model,
                device,
            ),
            "flow-neural-zoomout": lambda: compute_p2p_with_flows_composition_neural_zoomout(
                source_path,
                target_path,
                source_features,
                target_features,
                source_model,
                target_model,
                device,
            ),
            "fmaps": lambda: compute_p2p_with_fmaps(
                source_path, target_path, source_features, target_features
            ),
            "fmap-zoomout": lambda: compute_p2p_with_fmap_zoomout(
                source_path, target_path, source_features, target_features
            ),
            "fmap-neural-zoomout": lambda: compute_p2p_with_fmap_neural_zoomout(
                source_path, target_path, source_features, target_features
            ),
            "knn-in-gauss": lambda: compute_p2p_with_inverted_flows_in_gauss(
                source_features, target_features, source_model, target_model
            ),
            "ndp-landmarks": lambda: ndp_with_ldmks(
                source_path,
                target_path,
                source_landmarks=source_landmarks,
                target_landmarks=target_landmarks,
            ),
            "fmap-wks": lambda: compute_p2p_with_fmaps_wks(
                source_path,
                target_path,
                source_landmarks=source_landmarks,
                target_landmarks=target_landmarks,
            ),
        }
    elif matching_methods == "baselines":
        return {
            "knn": lambda: compute_p2p_with_knn(source_features, target_features),
            "ot": lambda: compute_p2p_with_ot(source_features, target_features),
            "flow": lambda: compute_p2p_with_flows_composition(
                source_features, target_features, source_model, target_model, device
            ),
            "fmaps": lambda: compute_p2p_with_fmaps(
                source_path, target_path, source_features, target_features
            ),
            "fmap-zoomout": lambda: compute_p2p_with_fmap_zoomout(
                source_path, target_path, source_features, target_features
            ),
            "fmap-neural-zoomout": lambda: compute_p2p_with_fmap_neural_zoomout(
                source_path, target_path, source_features, target_features
            ),
            "knn-in-gauss": lambda: compute_p2p_with_inverted_flows_in_gauss(
                source_features, target_features, source_model, target_model
            ),
            "ndp-landmarks": lambda: ndp_with_ldmks(
                source_path,
                target_path,
                source_landmarks=source_landmarks,
                target_landmarks=target_landmarks,
            ),
            "fmap-wks": lambda: compute_p2p_with_fmaps_wks(
                source_path,
                target_path,
                source_landmarks=source_landmarks,
                target_landmarks=target_landmarks,
            ),
        }


    elif matching_methods == "baselines-no-zoomout":
        return {
            "knn": lambda: compute_p2p_with_knn(source_features, target_features),
            "ot": lambda: compute_p2p_with_ot(source_features, target_features),
            "flow": lambda: compute_p2p_with_flows_composition(
                source_features, target_features, source_model, target_model, device
            ),
            "fmaps": lambda: compute_p2p_with_fmaps(
                source_path, target_path, source_features, target_features
            ),
            "knn-in-gauss": lambda: compute_p2p_with_inverted_flows_in_gauss(
                source_features, target_features, source_model, target_model
            ),
            "ndp-landmarks": lambda: ndp_with_ldmks(
                source_path,
                target_path,
                source_landmarks=source_landmarks,
                target_landmarks=target_landmarks,
            ),
            "fmap-wks": lambda: compute_p2p_with_fmaps_wks(
                source_path,
                target_path,
                source_landmarks=source_landmarks,
                target_landmarks=target_landmarks,
            ),
        }
    elif matching_methods == "zoomout":
        return {
            "flow-zoomout": lambda: compute_p2p_with_flows_composition_zoomout(
                source_path,
                target_path,
                source_features,
                target_features,
                source_model,
                target_model,
                device,
            ),
            "flow-neural-zoomout": lambda: compute_p2p_with_flows_composition_neural_zoomout(
                source_path,
                target_path,
                source_features,
                target_features,
                source_model,
                target_model,
                device,
            ),
            "fmap-zoomout": lambda: compute_p2p_with_fmap_zoomout(
                source_path, target_path, source_features, target_features
            ),
            "fmap-neural-zoomout": lambda: compute_p2p_with_fmap_neural_zoomout(
                source_path, target_path, source_features, target_features
            ),
        }
    else:
        raise ValueError(f"Unknown matching methods option: {matching_methods}")


def compute_geodesic_distance(
    source_mesh: trimesh.Trimesh,
    target_mesh: trimesh.Trimesh,
    data_path: DataPath,
) -> np.ndarray:
    os.makedirs(data_path.dists_path, exist_ok=True)
    dist_cache_path = os.path.join(data_path.dists_path, f"{target}_dists.npy")

    if os.path.exists(dist_cache_path):
        dist = np.load(dist_cache_path)
    else:
        if len(target_mesh.faces) > 0:
            mesh_gf = TriangleMesh(target_mesh.vertices, np.array(target_mesh.faces))
            heat = HeatDistanceMetric.from_registry(mesh_gf)
            dist = heat.dist_matrix()
        else:
            solver = pp3d.PointCloudHeatSolver(vertices)
            distances = []
            for idx in range(len(vertices)):
                distances.append(solver.compute_distance(idx))
            dist = np.array(distances)
        np.save(dist_cache_path, dist)

    return dist


def run_matching_methods_parallel(
    matching_methods,
    target_points,
    source_mesh,
    target_mesh,
    dists,
    source_corr=None,
    target_corr=None,
    max_workers=10,
):
    results = {}

    def wrapper(name, fn):
        return name, run_matching_methods(
            {name: fn},
            target_points,
            source_mesh,
            target_mesh,
            dists,
            source_corr,
            target_corr,
        )[name]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(wrapper, name, fn): name for name, fn in matching_methods.items()}
        for future in as_completed(futures):
            name, res = future.result()
            results[name] = res

    return results


def run_matching_methods(
    matching_methods: Dict[str, Callable[[], torch.Tensor]],
    target_points: torch.Tensor,
    source_mesh: trimesh.Trimesh,
    target_mesh: trimesh.Trimesh,
    dists: np.ndarray,
    source_corr=None,
    target_corr=None,
) -> Dict[str, MatchingResult]:
    """Run all P2P strategies and compute matched points + errors."""
    results = {}
    for name, func in matching_methods.items():
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        p2p, elapsed = func()

        max_euclidean_error = torch.cdist(target_points, target_points).max().item()

        if source_corr is None and target_corr is None:
            matched_points = target_points[p2p]
            euclidean_error = torch.norm(matched_points - target_points, dim=-1).mean().item() / max_euclidean_error
            geodesic_error = compute_geodesic_error(dists, p2p, None, None) / dists.max()
            dirichlet_energy = compute_dirichlet_energy(source_mesh, target_mesh, p2p)
            coverage = compute_coverage(p2p, len(target_mesh.vertices))

        else:
            tqdm.write(f"Applying correspondences alignment for {name}")
            matched_points = target_points[p2p]
            target_points_corr = target_points
            matched_points = matched_points[source_corr]
            target_points_corr = target_points_corr[target_corr]

            euclidean_error = torch.norm(matched_points - target_points_corr, dim=-1).mean().item() / max_euclidean_error
            geodesic_error = compute_geodesic_error(dists, p2p, source_corr, target_corr) / dists.max()
            dirichlet_energy = compute_dirichlet_energy(source_mesh, target_mesh, p2p)
            coverage = compute_coverage(p2p, len(target_mesh.vertices))

        results[name] = MatchingResult(
            indices=p2p,
            matched_points=matched_points,
            euclidean_error=euclidean_error,
            geodesic_error=geodesic_error,
            dirichlet_energy=dirichlet_energy,
            coverage=coverage,
            elapsed=elapsed,
        )

    return results


def log_results(source: str, target: str, results: Dict[str, MatchingResult]) -> None:
    """tqdm.write error metrics nicely formatted."""
    tqdm.write(f"> Evaluation results for {source} -> {target}:")
    for name, res in results.items():
        tqdm.write(
            f"  > {name:<20}: euclidean_error {res.euclidean_error:.4f} | geodesic_error {res.geodesic_error:.4f} | dirichlet_energy={res.dirichlet_energy:.4f} | coverage={res.coverage:.4f} | elapsed={res.elapsed:.2f}s"
        )


def plot_results(
    source_points: torch.Tensor,
    source: str,
    target: str,
    results: Dict[str, MatchingResult],
    output_dir: str,
    plot_html: bool,
    plot_png: bool,
    max_points: int = 100_000,
) -> None:
    """Plot correspondences for each method."""
    source_points = source_points[:max_points]
    for name, res in results.items():
        start_end_subplot(
            source_points,
            res.matched_points[:max_points],
            plots_path=str(output_dir),
            run_name=f"{name}-{source}-{target}",
            show=False,
            html=plot_html,
            png=plot_png,
        )


def get_geodesic_dists(
    mesh: trimesh.Trimesh,
    target: str,
    data_path: DataPath
) -> np.ndarray:
    """
    Compute or load geodesic distance matrix for a target mesh.

    Args:
        mesh (trimesh.Trimesh): The mesh to compute distances on.
        target_name (str): Identifier for caching (e.g. filename stem).
        data_path (DataPath): Dataclass containing paths for caching.
    Returns:
        np.ndarray: Geodesic distance matrix of shape (n_vertices, n_vertices).
    """
    os.makedirs(data_path.dists_path, exist_ok=True)
    dist_cache_path = os.path.join(data_path.dists_path, f"{target}_dists.npy")

    if os.path.exists(dist_cache_path):
        dist = np.load(dist_cache_path)
    else:
        if len(mesh.faces) > 0:
            mesh_gf = TriangleMesh(mesh.vertices, np.array(mesh.faces))
            heat = HeatDistanceMetric.from_registry(mesh_gf)
            dist = heat.dist_matrix()
        else:
            solver = pp3d.PointCloudHeatSolver(mesh.vertices)
            distances = []
            for idx in range(len(mesh.vertices)):
                distances.append(solver.compute_distance(idx))
            dist = np.array(distances)
        np.save(dist_cache_path, dist)

    return dist


def process_pair(
    source: str,
    target: str,
    source_rep: str,
    target_rep: str,
    device: str,
    mesh_baseline: bool,
    plot_html: bool,
    plot_png: bool,
    all_methods: str,
    features_normalization: str,
    data_path: DataPath,
) -> pd.DataFrame:
    """
    Main pipeline to process a source-target pair, evaluate multiple P2P methods,
    log and plot results, and return a DataFrame of errors.
    """

    source_features, source_points, source_model, source_mesh = process_element(
        source, source_rep, device, mesh_baseline, features_normalization, data_path
    )
    target_features, target_points, target_model, target_mesh = process_element(
        target, target_rep, device, mesh_baseline, features_normalization, data_path
    )

    source_geo_dists = get_geodesic_dists(source_mesh, source, data_path)
    target_geo_dists = get_geodesic_dists(target_mesh, target, data_path)

    if data_path.corr_path is not None:
        source_corr = (
            np.loadtxt(Path(data_path.corr_path, source + ".vts")).astype(int) - 1
        )
        target_corr = (
            np.loadtxt(Path(data_path.corr_path, target + ".vts")).astype(int) - 1
        )
    else:
        source_corr = np.arange(len(source_mesh.vertices))
        target_corr = np.arange(len(target_mesh.vertices))

    source_landmarks = source_corr[data_path.landmarks]
    target_landmarks = target_corr[data_path.landmarks]

    # TODO: Move target and source in dataclasses
    matching_methods = get_matching_methods(
        source_features=source_features,
        target_features=target_features,
        source_path=Path(data_path.dataset_path, source + data_path.dataset_extension),
        target_path=Path(data_path.dataset_path, target + data_path.dataset_extension),
        source_model=source_model,
        target_model=target_model,
        source_landmarks=source_landmarks,
        target_landmarks=target_landmarks,
        device=device,
        matching_methods=all_methods,
    )

    if data_path.corr_path is not None:
        source_corr = np.loadtxt(Path(data_path.corr_path, source + ".vts")).astype(int) - 1
        target_corr = np.loadtxt(Path(data_path.corr_path, target + ".vts")).astype(int) - 1
        results = run_matching_methods(
            matching_methods=matching_methods,
            target_points=target_points,
            source_mesh=source_mesh,
            target_mesh=target_mesh,
            dists=target_geo_dists,
            source_corr=source_corr,
            target_corr=target_corr,
        )
    else:
        tqdm.write("No correspondence path provided")
        results = run_matching_methods(
            matching_methods=matching_methods,
            target_points=target_points,
            source_mesh=source_mesh,
            target_mesh=target_mesh,
            dists=target_geo_dists,
            source_corr=None,
            target_corr=None,
        )

    log_results(source, target, results)
    plot_results(
        source_points=source_points,
        source=source,
        target=target,
        results=results,
        output_dir=str(data_path.output_dir),
        plot_html=plot_html,
        plot_png=plot_png,
        max_points=100_000,
    )

    rows = []
    for name, res in results.items():
        rows.append(
            {
                "source": source,
                "target": target,
                "method": name,
                "euclidean_error": res.euclidean_error,
                "geodesic_error": res.geodesic_error,
                "dirichlet": res.dirichlet_energy,
                "coverage": res.coverage,
                "elapsed": res.elapsed,
            }
        )
    df = pd.DataFrame(rows)

    return df


def plot_matching_error(err_flow_composition, map_err_knn, output_dir):
    shapes = sorted(
        set(
            [k[0] for k in err_flow_composition.keys()]
            + [k[1] for k in err_flow_composition.keys()]
        )
    )
    plt.figure(figsize=(14, 8))
    color_map = plt.get_cmap("tab20")
    for idx, shape in enumerate(shapes):
        color = color_map(idx % 20)
        x = []
        y_flow = []
        y_knn = []
        for (source, target), err_flow in err_flow_composition.items():
            if source == shape:
                x.append(target)
                y_flow.append(err_flow)
                y_knn.append(map_err_knn[(source, target)])
        if x:
            plt.plot(
                x,
                y_flow,
                marker="o",
                label=f"Flow Inversion Error ({shape})",
                color=color,
            )
            plt.plot(
                x,
                y_knn,
                marker="x",
                label=f"KNN Error ({shape})",
                color=color,
                linestyle="--",
            )
    plt.title("Errors for all shapes as source")
    plt.xlabel("Target Shape")
    plt.ylabel("Error")
    plt.xticks(rotation=45)
    plt.tight_layout(rect=[0, 0, 0.75, 1])
    plt.savefig(f"{output_dir}/plot_errors_all_shapes.png", bbox_inches="tight")
    plt.close()


def get_geodesic_dists(
    mesh: trimesh.Trimesh,
    target: str,
    data_path: DataPath
) -> np.ndarray:
    """
    Compute or load geodesic distance matrix for a target mesh.

    Args:
        mesh (trimesh.Trimesh): The mesh to compute distances on.
        target_name (str): Identifier for caching (e.g. filename stem).
        cache_dir (str): Directory where cached distance matrices are stored.
        device (str): Torch device ("cpu" or "cuda").

    Returns:
        np.ndarray: Geodesic distance matrix of shape (n_vertices, n_vertices).
    """
    os.makedirs(data_path.dists_path, exist_ok=True)
    dist_cache_path = os.path.join(data_path.dists_path, f"{target}_dists.npy")

    if os.path.exists(dist_cache_path):
        dist = np.load(dist_cache_path)
    else:
        vertices = np.array(mesh.vertices)
        if len(mesh.faces) > 0:
            mesh_gf = TriangleMesh(vertices, np.array(mesh.faces))
            heat = HeatDistanceMetric.from_registry(mesh_gf)
            dist = heat.dist_matrix()
        else:
            solver = pp3d.PointCloudHeatSolver(vertices)
            distances = []
            for idx in range(len(vertices)):
                distances.append(solver.compute_distance(idx))
            dist = np.array(distances)
        np.save(dist_cache_path, dist)

    return dist


def main(args):
    with open(args.config, "r") as f:
        config = json.load(f)

    device = config["device"]
    torch.cuda.set_device(device)
    mesh_baseline = args.mesh_baseline

    if args.faust:
        dataset = "FAUST"
        targets = get_targets_faust(args)
    elif args.smal:
        dataset = "SMAL"
        targets = get_targets_smal(Path(config["matching_config"][dataset]["flows_path"]))
    else:
        raise ValueError("Please specify either --faust or --smal")

    if targets is None or len(targets) == 0:
        raise ValueError("No targets found to process.")
    tqdm.write(f"Running {dataset} experiments")
    tqdm.write(f"Found {len(targets)} targets: {targets}")

    output_dir = Path(f"./out/matching/{dataset.lower()}/matching-{args.source_rep}-{args.target_rep}-{args.matching_run_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    tqdm.write(f"Output directory: {output_dir}")

    data_path = DataPath(
        output_dir=output_dir,
        landmarks=config["matching_config"][dataset]["landmarks"],
        dataset_path=Path(config["matching_config"][dataset]["dataset_path"]),
        dists_path=Path(config["matching_config"][dataset]["dists_path"]),
        features_path=Path(config["matching_config"][dataset]["features_path"]),
        dataset_extension=config["matching_config"][dataset]["dataset_extension"],
        flows_path=Path(config["matching_config"][dataset]["flows_path"]),
        flows_SDFs_path=Path(config["matching_config"][dataset]["flows_SDFs_path"]),
        sdf_path=Path(config["matching_config"][dataset]["SDFs_path"]),
        corr_path=Path(config["matching_config"][dataset]["corr_path"]) if "corr_path" in config["matching_config"][dataset] else None
    )

    results = []
    times = []
    results_file = Path(output_dir, "matching_results.csv")

    if args.same:
        pairs = [(t, t) for t in targets]
    else:
        pairs = [(s, t) for s in targets for t in targets if s != t]

    for source, target in tqdm(pairs, desc="Processing shape pairs", unit="pair", dynamic_ncols=True):
        tqdm.write(f"Processing {source} -> {target}")
        start_time = time.perf_counter()
        df = process_pair(
            source=source,
            target=target,
            source_rep=args.source_rep,
            target_rep=args.target_rep,
            device=device,
            mesh_baseline=mesh_baseline,
            plot_html=args.plot_html,
            plot_png=args.plot_png,
            all_methods=args.matching_methods,
            features_normalization=args.features_normalization,
            data_path=data_path,
        )
        elapsed_time = time.perf_counter() - start_time
        tqdm.write(f"Time taken for {source} -> {target}: {elapsed_time:.2f} seconds")
        times.append(elapsed_time)

        df.to_csv(results_file, mode="a", header=not results_file.exists(), index=False)
        results.append(df)

    # Save and tqdm.write average time
    times_sec = [t.total_seconds() if hasattr(t, "total_seconds") else float(t) for t in times]
    avg_time = sum(times_sec) / len(times_sec)
    tqdm.write(f"Average time for all pairs: {avg_time:.2f} seconds")
    with open(Path(output_dir, "timing_results.txt"), "w") as f:
        for i, t in enumerate(times_sec):
            f.write(f"Pair {i + 1}: {t:.2f} seconds\n")
        f.write(f"\nAverage time: {avg_time:.2f} seconds\n")

    results_df = pd.concat(results, ignore_index=True)
    results_df.to_csv(Path(output_dir, "matching_results_completed.csv"), index=False)

    tqdm.write("Average errors across all pairs:")
    avg_metrics = df.groupby("method")[
        ["euclidean_error", "geodesic_error", "dirichlet", "coverage", "elapsed"]
    ].mean()
    tqdm.write(avg_metrics.to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Matching experiments for SDFs and meshes"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use (e.g., 'cpu', 'cuda:0, 'cuda:1')",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to the config file",
        required=True,
        default="config.json",
    )
    parser.add_argument(
        "--faust",
        action="store_true",
        help="Run matching methods on FAUST dataset",
        default=False,
    )
    parser.add_argument(
        "--smal",
        action="store_true",
        help="Run matching methods on SMAL dataset",
        default=False,
    )
    parser.add_argument(
        "--mesh_baseline",
        action="store_true",
        help="Use the mesh vertices features instead of randomly sampled points",
        default=False,
    )
    parser.add_argument(
        "--plot_png",
        action="store_true",
        help="Save a PNG plot of the matching",
        default=False,
    )
    parser.add_argument(
        "--plot_html",
        action="store_true",
        help="Save an HTML plot of the matching",
        default=False,
    )
    parser.add_argument(
        "--source_rep",
        type=str,
        default="mesh",
        help="Representation of the first element in the pair (e.g., 'mesh', 'sdf')",
    )
    parser.add_argument(
        "--target_rep",
        type=str,
        default="mesh",
        help="Representation of the first element in the pair (e.g., 'mesh', 'sdf')",
    )
    parser.add_argument(
        "--same",
        action="store_true",
        help="Match the same shape with different representations",
        default=False,
    )
    parser.add_argument(
        "--geo_error",
        action="store_true",
        help="If true along side mesh_baseline and source_rep 'sdf', for each landmark plot the difference between the true geodesic distance and the Dijkastra approximation",
        default=False,
    )
    parser.add_argument(
        "--matching_run_name",
        type=str,
        help="Name of the run to append at the end of the output directory",
        default="",
    )
    parser.add_argument(
        "--matching_methods",
        type=str,
        default="fast",
        help="Which matching methods to use: 'fast' (knn, flow), 'hungarian' (knn, hungarian, flow, flow-hungarian), 'all' (knn, hungarian, lapjv, flow, flow-hungarian, flow-lapjv)",
    )
    parser.add_argument(
        "--features_normalization",
        default="none",
        type=str,
        help="Normalization to apply to the features: none, 0_1_indipendent, 0_1_global, 0_center_indipendent, 0_center_global",
    )

    args = parser.parse_args()
    if args.config:
        with open(args.config, "r") as f:
            config = json.load(f)
        for key, value in config.items():
            if not hasattr(args, key):
                continue
            setattr(args, key, value)

    if args.matching_run_name == "":
        args.matching_run_name = input("Enter a run name: ")

    tqdm.write("------------------------------------------")
    for arg in vars(args):
        tqdm.write(f"{arg}: {getattr(args, arg)}")
    tqdm.write("------------------------------------------")

    main(args)
