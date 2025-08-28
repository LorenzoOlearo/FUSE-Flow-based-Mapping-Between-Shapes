import os
from pathlib import Path
from typing import List
import argparse
import csv
import numpy as np
import torch
import trimesh
import matplotlib.colors as mcolors

from typing import Callable, Dict
from dataclasses import dataclass
import pandas as pd

from util.matching_utils import (
    compute_p2p_with_flows_composition,
    compute_p2p_with_flows_composition_hungarian,
    compute_p2p_with_flows_composition_lapjv,
    compute_p2p_with_knn_gauss,
    compute_p2p_with_knn,
    compute_p2p_with_lapjv,
    compute_p2p_with_fmaps,
    compute_p2p_with_hungarian,
)
from model.models import FMCond
from model.networks import MLP
from util.plot import start_end_subplot
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from util.plot import plot_points

SDFs_PATH = Path('./out/SDFs')
FAUST_PATH = Path('./data/MPI-FAUST/training/registrations/')
N_LANDMARKS = 5
FLOWS_PATH = Path('./out/flows')
FLOWS_VERTEX_PATH = Path('./out/flows_vertex')


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


def is_in_range(name):
    try:
        num = int(''.join(filter(str.isdigit, name)))
        return 80 <= num <= 100
    except ValueError:
        return False


def get_targets() -> List[str]:
    """Determine which targets to process."""
    # targets = [
    #     f.name for f in SDFs_PATH.iterdir() if f.is_dir() and
    #     any(child.name.endswith('-features.txt') for child in f.iterdir()) and
    #     (FLOWS_PATH / f.name / 'checkpoint-9999.pth').is_file() and
    #     is_in_range(f.name)
    # ]

    targets = []
    for i in range(80, 100):
        target = f"tr_reg_{i:03d}"
        targets.append(target)

    print(f"Processing all targets: {targets}")
    return targets


def process_element(element: str, representation: str, device: str, mesh_baseline: bool):
    mesh_path = Path(FAUST_PATH, element + '.ply')
    mesh = trimesh.load(mesh_path, process=False)
    mesh = normalize_mesh(mesh)
    model = FMCond(channels=N_LANDMARKS, network=MLP(channels=N_LANDMARKS).to(device))

    if representation == 'mesh':
        features_path = Path(FLOWS_VERTEX_PATH, element, f'features.txt')
        points = torch.tensor(mesh.vertices.astype(np.float32)).to(device)
        features = torch.tensor(np.loadtxt(features_path).astype(np.float32)).to(device)
        model.load_state_dict(torch.load(Path(FLOWS_VERTEX_PATH, element, 'checkpoint-9999.pth'), weights_only=False)['model'], strict=True)
    elif representation == 'sdf':
        print(f"Processing {element} with SDF representation")
        if mesh_baseline is False:
            features_path = Path(SDFs_PATH, element, f'{element}-sdf-dijkstra-features.txt')
            points_path = Path(SDFs_PATH, element, f'{element}-sdf-sampled-points.txt')
            points = torch.tensor(np.loadtxt(points_path).astype(np.float32)).to(device)
        else:
            print(f"Using mesh baseline for {element}")
            features_path = Path(SDFs_PATH, element, f'{element}-sdf-mesh-dists-projected-interpolated.txt')
            points = torch.tensor(mesh.vertices.astype(np.float32)).to(device)
        features = torch.tensor(np.loadtxt(features_path).astype(np.float32)).to(device)
        model.load_state_dict(torch.load(Path(FLOWS_PATH, element, 'checkpoint-9999.pth'), weights_only=False)['model'], strict=True)

    model.to(device)
    model.eval()

    return features, points, model


@dataclass
class P2PResult:
    """Holds the output of a P2P method."""
    indices: torch.Tensor
    matched_points: torch.Tensor
    error: float


def prepare_elements(
    source: str, target: str, source_rep: str, target_rep: str, device: str, mesh_baseline: bool
):
    """Extract features, points, and models for source and target."""
    source_features, source_points, source_model = process_element(source, source_rep, device, mesh_baseline)
    target_features, target_points, target_model = process_element(target, target_rep, device, mesh_baseline)

    return source_features, source_points, source_model, target_features, target_points, target_model


def get_matching_methods(
    source_features, target_features, source_model, target_model, device: str, all_methods: bool = False
) -> Dict[str, Callable[[], torch.Tensor]]:
    """Return mapping of strategy names to their compute functions."""

    if all_methods is False:
        return {
            "knn": lambda: compute_p2p_with_knn(source_features, target_features),
            "flow": lambda: compute_p2p_with_flows_composition(
                source_features, target_features, source_model, target_model, device
            ),
        }
    else:
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


def run_matching_methods(
    strategies: Dict[str, Callable[[], torch.Tensor]], target_points: torch.Tensor
) -> Dict[str, P2PResult]:
    """Run all P2P strategies and compute matched points + errors."""
    results = {}
    for name, func in strategies.items():
        indices = func()
        matched_points = target_points[indices]
        error = torch.norm(matched_points - target_points, dim=-1).mean().item()
        results[name] = P2PResult(indices=indices, matched_points=matched_points, error=error)
    return results


def log_results(source: str, target: str, results: Dict[str, P2PResult]) -> None:
    """Print error metrics nicely formatted."""
    print(f"> Evaluation results for {source} -> {target}:")
    for name, res in results.items():
        print(f"  > {name:<20}: {res.error:.4f}")


def plot_results(
    source_points: torch.Tensor,
    source: str,
    target: str,
    results: Dict[str, P2PResult],
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
    all_methods: bool = False,
) -> pd.DataFrame:
    """
    Main pipeline to process a source-target pair, evaluate multiple P2P methods,
    log and plot results, and return a DataFrame of errors.
    """
    source_features, source_points, source_model, target_features, target_points, target_model = prepare_elements(source, target, source_rep, target_rep, device, mesh_baseline)

    matching_methods = get_matching_methods(source_features, target_features, source_model, target_model, device, all_methods)
    results = run_matching_methods(matching_methods, target_points)
    
    log_results(source, target, results)
    plot_results(source_points, source, target, results, output_dir, plot_html, plot_png)

    row = {"source": source, "target": target}
    row.update({name: res.error for name, res in results.items()})
    df = pd.DataFrame([row])

    if mesh_baseline and geo_error:
        geo = np.loadtxt(Path(FLOWS_VERTEX_PATH, source, 'features.txt'))
        mesh = trimesh.load(Path(FAUST_PATH, source + '.ply'), process=False)
        mesh = normalize_mesh(mesh)
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
    device = 'cuda:1'
    torch.cuda.set_device(device)
    mesh_baseline = args.mesh_baseline

    output_dir = Path(f'./out/matching/matching-{args.source_rep}-{args.target_rep}-{args.run_name}')
    os.makedirs(output_dir, exist_ok=True)

    targets = get_targets()

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
                args.all_methods
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
                        args.all_methods
                    )
                    results.append(df)

    results_df = pd.concat(results, ignore_index=True)

    print("Average errors across all pairs:")
    for col in results_df.columns:
        if col not in ['source', 'target']:
            avg_error = results_df[col].mean()
            print(f"  > {col:<20}: {avg_error:.4f}")


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
    parser.add_argument('--all_methods', action='store_true', help="Run all matching methods, default is only knn and flow inversion because the others are too slow", default=False)

    args = parser.parse_args()
    if args.run_name == "":
        args.run_name = input("Enter a run name: ")

    print("------------------------------------------")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print("------------------------------------------")

    main(args)
