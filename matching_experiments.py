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
from typing import Callable, Dict
from dataclasses import dataclass
import pandas as pd

from util.matching_utils import (
    compute_p2p_with_flows_composition,
    compute_p2p_with_flows_composition_hungarian,
    compute_p2p_with_flows_composition_lapjv,
    compute_p2p_with_inverted_flows_in_gauss,
    compute_p2p_with_inverted_flows_in_gauss_uniformed,
    compute_p2p_with_knn,
    compute_p2p_with_lapjv,
    compute_p2p_with_fmaps,
    compute_p2p_with_hungarian,
)

from util.metrics import compute_dirichlet_energy, compute_coverage

from model.models import FMCond
from model.networks import MLP
from util.plot import start_end_subplot
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from util.plot import plot_points

N_LANDMARKS = 5
FAUST_PATH = Path('./data/MPI-FAUST/training/registrations/')
FLOWS_PATH = Path('./out/flows/faust/faust-norm-0-1-indipendent/')
FLOWS_SDF_PATH = Path('./out/flows_vertex_SDFs/')
SDFs_PATH = Path('./out/SDFs')


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
        num = int(''.join(filter(str.isdigit, name)))
        return 80 <= num <= 100
    except ValueError:
        return False


def get_targets(args) -> List[str]:
    """Determine which targets to process."""
    # targets = [
    #     f.name for f in SDFs_PATH.iterdir() if f.is_dir() and
    #     any(child.name.endswith('-features.txt') for child in f.iterdir()) and
    #     (FLOWS_PATH / f.name / 'checkpoint-9999.pth').is_file() and
    #     is_in_range(f.name)
    # ]

    targets = []
    for i in range(80, 100):
        if args.source_rep == 'sdf' or args.target_rep == 'sdf':
            if i == 86 or i == 87 or i == 96 or i == 97:
                print(f"Skipping topologically incorrect shape tr_reg_{i:03d}")
                continue
        target = f"tr_reg_{i:03d}"
        targets.append(target)

    print(f"Processing all targets: {targets}")
    return targets


def normalize_features(features, method):
    """Normalize features based on the specified method."""
    if method == "none":
        print("No normalization applied to features.")
        return features
    elif method == "0_1_indipendent":
        # Normalize each feature channel to [0, 1]
        min_vals = features.min(dim=0).values
        max_vals = features.max(dim=0).values
        normalized = (features - min_vals) / (max_vals - min_vals + 1e-8)
        print(f"Feature normalization (0_1_indipendent): min {normalized.min(dim=0).values.tolist()}, max {normalized.max(dim=0).values.tolist()}")
        return normalized
    elif method == "0_1_global":
        # Normalize all features to [0, 1] based on global min and max
        min_val = features.min()
        max_val = features.max()
        normalized = (features - min_val) / (max_val - min_val + 1e-8)
        print(f"Feature normalization (0_1_global): min {normalized.min(dim=0).values.tolist()}, max {normalized.max(dim=0).values.tolist()}")
        return normalized
    elif method == "0_center_indipendent":
        # Center each feature channel around 0 and scale to [-1, 1]
        mean_vals = features.mean(dim=0)
        max_devs = torch.max(torch.abs(features - mean_vals), dim=0).values
        normalized = (features - mean_vals) / (max_devs + 1e-8)
        print(f"Feature normalization (0_center_indipendent): min {normalized.min(dim=0).values.tolist()}, max {normalized.max(dim=0).values.tolist()}")
        return normalized
    elif method == "0_center_global":
        # Center all features around 0 and scale to [-1, 1] based on global max deviation
        mean_val = features.mean()
        max_dev = torch.max(torch.abs(features - mean_val))
        normalized = (features - mean_val) / (max_dev + 1e-8)
        print(f"Feature normalization (0_center_global): min {normalized.min(dim=0).values.tolist()}, max {normalized.max(dim=0).values.tolist()}")
        return normalized
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def process_element(element: str, representation: str, device: str, mesh_baseline: bool, features_normalization: str):
    mesh_path = Path(FAUST_PATH, element + '.ply')
    mesh = trimesh.load(mesh_path, process=False)
    mesh = normalize_mesh_08(mesh)
    model = FMCond(channels=N_LANDMARKS, network=MLP(channels=N_LANDMARKS).to(device))

    if representation == 'mesh':
        features_path = Path(FLOWS_PATH, element, f'vertex-geodesics.txt')
        points = torch.tensor(mesh.vertices.astype(np.float32)).to(device)
        features = torch.tensor(np.loadtxt(features_path).astype(np.float32)).to(device)
        model.load_state_dict(torch.load(Path(FLOWS_PATH, element, 'checkpoint-9999.pth'), weights_only=False)['model'], strict=True)
    elif representation == 'sdf':
        print(f"Processing {element} with SDF representation")
        features_path = Path(SDFs_PATH, element, f'{element}-sdf-projected-vertex-dists.txt')
        points = torch.tensor(mesh.vertices.astype(np.float32)).to(device)
        # if mesh_baseline is False:
        #     features_path = Path(SDFs_PATH, element, f'{element}-sdf-dijkstra-features.txt')
        #     points_path = Path(SDFs_PATH, element, f'{element}-sdf-sampled-points.txt')
        #     points = torch.tensor(np.loadtxt(points_path).astype(np.float32)).to(device)
        # else:
        #     print(f"Using mesh baseline for {element}")
        #     features_path = Path(SDFs_PATH, element, f'{element}-sdf-mesh-dists-projected-interpolated.txt')
        #     # features_path = Path(SDFs_PATH, element, f'{element}-sdf-extracted-mesh-dists.txt')   # Analitic SDF features
        #     points = torch.tensor(mesh.vertices.astype(np.float32)).to(device)
        features = torch.tensor(np.loadtxt(features_path).astype(np.float32)).to(device)
        model.load_state_dict(torch.load(Path(FLOWS_SDF_PATH, element, 'checkpoint-9999.pth'), weights_only=False)['model'], strict=True)

    features = normalize_features(features, features_normalization)

    model.to(device)
    model.eval()

    return features, points, model, mesh


@dataclass
class MatchingResult:
    """Holds the output of a P2P method."""
    indices: torch.Tensor
    matched_points: torch.Tensor
    euclidean_error: float
    dirichlet_energy: float
    coverage: float
    elapsed: float


def get_matching_methods(
    source_features, target_features, source_model, target_model, device: str, matching_methods: str
) -> Dict[str, Callable[[], torch.Tensor]]:
    """Return mapping of strategy names to their compute functions."""

    if matching_methods == "fast":
        return {
            "knn": lambda: compute_p2p_with_knn(source_features, target_features),
            "flow": lambda: compute_p2p_with_flows_composition(
                source_features, target_features, source_model, target_model, device
            ),
        }
    elif matching_methods == "hungarian":
        return {
            "knn": lambda: compute_p2p_with_knn(source_features, target_features),
            "hungarian": lambda: compute_p2p_with_hungarian(source_features, target_features),
            "flow": lambda: compute_p2p_with_flows_composition(
                source_features, target_features, source_model, target_model, device
            ),
            "flow-hungarian": lambda: compute_p2p_with_flows_composition_hungarian(
                source_features, target_features, source_model, target_mode
            ),
        }
    elif matching_methods == "all":
        return {
            "knn": lambda: compute_p2p_with_knn(source_features, target_features),
            "hungarian": lambda: compute_p2p_with_hungarian(source_features, target_features),
            "lapjv": lambda: compute_p2p_with_lapjv(source_features, target_features),

            "flow": lambda: compute_p2p_with_flows_composition(
                source_features, target_features, source_model, target_model, device
            ),
            "flow-hungarian": lambda: compute_p2p_with_flows_composition_hungarian(
                source_features, target_features, source_model, target_model
            ),
            "flow-lapjv": lambda: compute_p2p_with_flows_composition_lapjv(
                source_features, target_features, source_model, target_model
            ),
        }
    else:
        raise ValueError(f"Unknown matching methods option: {matching_methods}")


def run_matching_methods(
    strategies: Dict[str, Callable[[], torch.Tensor]],
    target_points: torch.Tensor,
    source_mesh: trimesh.Trimesh,
    target_mesh: trimesh.Trimesh,
) -> Dict[str, MatchingResult]:
    """Run all P2P strategies and compute matched points + errors."""
    results = {}
    for name, func in strategies.items():
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        indices = func()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        matched_points = target_points[indices]
        euclidean_error = torch.norm(matched_points - target_points, dim=-1).mean().item()
        dirichlet_energy = compute_dirichlet_energy(source_mesh, target_mesh, indices)
        coverage = compute_coverage(indices, len(target_mesh.vertices))

        results[name] = MatchingResult(
            indices=indices,
            matched_points=matched_points,
            euclidean_error=euclidean_error,
            dirichlet_energy=dirichlet_energy,
            coverage=coverage,
            elapsed=elapsed,
        )

    return results


def log_results(source: str, target: str, results: Dict[str, MatchingResult]) -> None:
    """Print error metrics nicely formatted."""
    print(f"> Evaluation results for {source} -> {target}:")
    for name, res in results.items():
        print(f"  > {name:<20}: euclidean_error {res.euclidean_error:.4f} | dirichlet_energy={res.dirichlet_energy:.4f} | coverage={res.coverage:.4f} | elapsed={res.elapsed:.2f}s")


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


def process_pair(
    source: str,
    target: str,
    source_rep: str,
    target_rep: str,
    device: str,
    mesh_baseline: bool,
    plot_html: bool,
    plot_png: bool,
    geo_error: bool,
    output_dir: str,
    all_methods: bool,
    features_normalization: str
) -> pd.DataFrame:
    """
    Main pipeline to process a source-target pair, evaluate multiple P2P methods,
    log and plot results, and return a DataFrame of errors.
    """

    source_features, source_points, source_model, source_mesh = process_element(source, source_rep, device, mesh_baseline, features_normalization)
    target_features, target_points, target_model, target_mesh = process_element(target, target_rep, device, mesh_baseline, features_normalization)

    matching_methods = get_matching_methods(source_features, target_features, source_model, target_model, device, all_methods)
    results = run_matching_methods(matching_methods, target_points, source_mesh, target_mesh)
    
    log_results(source, target, results)
    plot_results(source_points, source, target, results, output_dir, plot_html, plot_png)

    rows = []
    for name, res in results.items():
        rows.append({
            "source": source,
            "target": target,
            "method": name,
            "euclidean_error": res.euclidean_error,
            "dirichlet": res.dirichlet_energy,
            "coverage": res.coverage,
            "elapsed": res.elapsed,
        })
    df = pd.DataFrame(rows)

    if mesh_baseline and geo_error:
        mesh = trimesh.load(Path(FAUST_PATH, source + '.ply'), process=False)
        mesh = normalize_mesh_08(mesh)
        vertex_geodesics = np.loadtxt(Path(FLOWS_PATH, source, 'features.txt'))
        # err_geo = np.linalg.norm(geo - source_features.cpu().numpy(), axis=1)

        err_geo = np.abs(geo - source_features.cpu().numpy())
        err_geo /= err_geo.max()

        for i in range(geo.shape[1]):
            plot_points(
                source_points.cpu().numpy(),
                distances=err_geo[:, i],
                title=f"landmark {i} at {source}: geo - sdf_geo",
                save_path=Path(output_dir, f'err-geo-{source}-{target}-{i}'),
                save_html=True,
                save_png=True,
                colorbar_title='geo_err',
                colormap='hot',
                range=(0, 1),
            )

            plot_geodesic_comparison(
                mesh.vertices,
                mesh.faces,
                true_dists=geo[:, i],
                dijkstra_dists= source_features[:, i].cpu().numpy(),
                save_path=Path(output_dir, f'geo-comparison-{source}-{target}-{i}')
            )

    return df


def plot_matching_error(err_flow_composition, map_err_knn, output_dir):
    shapes = sorted(set([k[0] for k in err_flow_composition.keys()] + [k[1] for k in err_flow_composition.keys()]))
    plt.figure(figsize=(14, 8))
    color_map = plt.get_cmap('tab20')
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
            plt.plot(x, y_flow, marker='o', label=f'Flow Inversion Error ({shape})', color=color)
            plt.plot(x, y_knn, marker='x', label=f'KNN Error ({shape})', color=color, linestyle='--')
    plt.title('Errors for all shapes as source')
    plt.xlabel('Target Shape')
    plt.ylabel('Error')
    plt.xticks(rotation=45)
    plt.tight_layout(rect=[0, 0, 0.75, 1])
    plt.savefig(f'{output_dir}/plot_errors_all_shapes.png', bbox_inches='tight')
    plt.close()


def main(args):
    device = 'cuda:0'
    torch.cuda.set_device(device)
    mesh_baseline = args.mesh_baseline

    output_dir = Path(f'./out/matching/matching-{args.source_rep}-{args.target_rep}-{args.run_name}')
    os.makedirs(output_dir, exist_ok=True)

    targets = get_targets(args)

    results = []
    if args.same:
        for target in targets:
            print(f"Matching the same shape {target} from {args.source_rep} to {args.target_rep}")
            df = process_pair(
                target, target,
                args.source_rep, args.target_rep,
                device, mesh_baseline,
                args.plot_html, args.plot_png,
                args.geo_error, output_dir,
                args.matching_methods,
                features_normalization=args.features_normalization
            )
            results.append(df)
    else:
        for source in targets:
            for target in targets:
                if source is not target:
                    print(f"Processing {source} -> {target}")
                    df = process_pair(
                        source, target,
                        args.source_rep, args.target_rep,
                        device, mesh_baseline,
                        args.plot_html, args.plot_png,
                        args.geo_error, output_dir,
                        args.matching_methods,
                        features_normalization=args.features_normalization
                    )
                    results.append(df)

    results_df = pd.concat(results, ignore_index=True)
    results_df.to_csv(Path(output_dir, 'matching_results.csv'), index=False)

    print("Average errors across all pairs:")
    avg_metrics = df.groupby('method')[['euclidean_error', 'dirichlet', 'coverage']].mean()
    print(avg_metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Matching experiments for SDFs and meshes")
    parser.add_argument('--mesh_baseline', action='store_true', help="Use the mesh vertices features instead of randomly sampled points", default=False)
    parser.add_argument('--plot_png', action='store_true', help="Save a PNG plot of the matching", default=False)
    parser.add_argument('--plot_html', action='store_true', help="Save an HTML plot of the matching", default=False)
    parser.add_argument('--source_rep', type=str, default='mesh', help="Representation of the first element in the pair (e.g., 'mesh', 'sdf')")
    parser.add_argument('--target_rep', type=str, default='mesh', help="Representation of the first element in the pair (e.g., 'mesh', 'sdf')")
    parser.add_argument('--same', action='store_true', help="Match the same shape with different representations", default=False)
    parser.add_argument('--geo_error', action='store_true', help="If true along side mesh_baseline and source_rep 'sdf', for each landmark plot the difference between the true geodesic distance and the Dijkastra approximation", default=False)
    parser.add_argument('--run_name', type=str, help="Name of the run to append at the end of the output directory", default="")
    parser.add_argument('--matching_methods', type=str, default='fast', help="Which matching methods to use: 'fast' (knn, flow), 'hungarian' (knn, hungarian, flow, flow-hungarian), 'all' (knn, hungarian, lapjv, flow, flow-hungarian, flow-lapjv)")
    parser.add_argument("--features_normalization", default="none", type=str, help="Normalization to apply to the features: none, 0_1_indipendent, 0_1_global, 0_center_indipendent, 0_center_global")

    args = parser.parse_args()
    if args.run_name == "":
        args.run_name = input("Enter a run name: ")

    print("------------------------------------------")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print("------------------------------------------")

    main(args)
