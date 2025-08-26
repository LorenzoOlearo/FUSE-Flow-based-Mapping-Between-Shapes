import os
from pathlib import Path
from typing import List
import argparse
import csv
import numpy as np
import torch
import trimesh
import matplotlib.colors as mcolors

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


def process_pair(source: str, target: str, source_rep: str, target_rep: str, device: str, mesh_baseline: bool, plot_html: bool, plot_png: bool, geo_error: bool, output_dir: str):
    source_features, source_points, source_model = process_element(source, source_rep, device, mesh_baseline)
    target_features, target_points, target_model = process_element(target, target_rep, device, mesh_baseline)

    p2p_flow = compute_p2p_with_flows_composition(source_features, target_features, source_model, target_model, device)
    p2p_flow_h = compute_p2p_with_flows_composition_hungarian(source_features, target_features, source_model, target_model)
    p2p_flow_lapjv = compute_p2p_with_flows_composition_lapjv(source_features, target_features, source_model, target_model)
    p2p_knn = compute_p2p_with_knn(source_features, target_features)
    p2p_h = compute_p2p_with_hungarian(source_features, target_features)
    p2p_lapjv = compute_p2p_with_lapjv(source_features, target_features)

    matched_points_flow = target_points[p2p_flow]
    matched_points_flow_h = target_points[p2p_flow_h]
    matched_points_flow_lapjv = target_points[p2p_flow_lapjv]
    matched_points_knn = target_points[p2p_knn]
    matched_points_h = target_points[p2p_h]
    matched_points_lapjv = target_points[p2p_lapjv]

    err_flow = torch.norm(matched_points_flow - target_points, dim=-1).mean().item()
    err_flow_h = torch.norm(matched_points_flow_h - target_points, dim=-1).mean().item()
    err_flow_lapjv = torch.norm(matched_points_flow_lapjv - target_points, dim=-1).mean().item()
    err_knn = torch.norm(matched_points_knn - target_points, dim=-1).mean().item()
    err_h = torch.norm(matched_points_h - target_points, dim=-1).mean().item()
    err_lapjv = torch.norm(matched_points_lapjv - target_points, dim=-1).mean().item()

    print(f"> KNN error:            {source} -> {target}: {err_knn:.4f}")
    print(f"> Hungarian error:      {source} -> {target}: {err_h:.4f}")
    print(f"> LapJV error:          {source} -> {target}: {err_lapjv:.4f}")
    print(f"> Flow inversion error: {source} -> {target}: {err_flow:.4f}")
    print(f"> Flow Hungarian error: {source} -> {target}: {err_flow_h:.4f}")
    print(f"> Flow LapJV error:     {source} -> {target}: {err_flow_lapjv:.4f}")

    source_points = source_points[:100000]
    target_points = target_points[:100000]

    start_end_subplot(
        source_points, matched_points_flow,
        plots_path=str(output_dir),
        run_name=f'flow-{source}-{target}',
        show=False,
        html=plot_html,
        png=plot_png,
    )

    start_end_subplot(
        source_points, matched_points_flow_lapjv,
        plots_path=str(output_dir),
        run_name=f'flow-lapjv-{source}-{target}',
        show=False,
        html=plot_html,
        png=plot_png,
    )

    start_end_subplot(
        source_points, matched_points_flow_h,
        plots_path=str(output_dir),
        run_name=f'flow-hungarian-{source}-{target}',
        show=False,
        html=plot_html,
        png=plot_png,
    )

    start_end_subplot(
        source_points, matched_points_lapjv,
        plots_path=str(output_dir),
        run_name=f'lapjv-{source}-{target}',
        show=False,
        html=plot_html,
        png=plot_png,
    )

    start_end_subplot(
        source_points, matched_points_knn,
        plots_path=str(output_dir),
        run_name=f'knn-{source}-{target}',
        show=False,
        html=plot_html,
        png=plot_png,
    )

    start_end_subplot(
        source_points, matched_points_h,
        plots_path=str(output_dir),
        run_name=f'hungarian-{source}-{target}',
        show=False,
        html=plot_html,
        png=plot_png,
    )

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


    return err_flow, err_knn, err_flow_h, err_flow_lapjv, err_h, err_lapjv


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
    err_flow_composition = {}
    err_flow_h_composition = {}
    err_flow_lapjv_composition = {}
    map_err_knn = {}
    err_hungarian = {}
    map_err_lapjv = {}

    if args.same:
        for target in targets:
            print(f"Matching the same shape {target} from {args.source_rep} to {args.target_rep}")
            err_flow, err_knn, err_flow_h, err_flow_lapjv, err_h, err_lapjv = process_pair(target, target, args.source_rep, args.target_rep, device, mesh_baseline, args.plot_html, args.plot_png, args.geo_error, output_dir)
            err_flow_composition[(target, target)] = err_flow
            map_err_knn[(target, target)] = err_knn
            map_err_knn[(target, target)] = err_knn
            err_flow_h_composition[(target, target)] = err_flow_h
            err_flow_lapjv_composition[(target, target)] = err_flow_lapjv
            err_hungarian[(target, target)] = err_h
            map_err_lapjv[(target, target)] = err_lapjv
    else:
        for source in targets:
            for target in targets:
                if source is not target:
                    print(f"Processing {source} -> {target}")
                    err_flow, err_knn, err_flow_h, err_flow_lapjv, err_h, err_lapjv = process_pair(source, target, args.source_rep, args.target_rep, device, mesh_baseline, args.plot_html, args.plot_png, args.geo_error, output_dir)
                    err_flow_composition[(source, target)] = err_flow
                    map_err_knn[(source, target)] = err_knn
                    err_flow_h_composition[(source, target)] = err_flow_h
                    err_flow_lapjv_composition[(source, target)] = err_flow_lapjv
                    err_hungarian[(source, target)] = err_h
                    map_err_lapjv[(source, target)] = err_lapjv

    print(f"Average KNN error:            {np.mean(list(map_err_knn.values())):.4f}")
    print(f"Average Hungarian error:      {np.mean(list(err_hungarian.values())):.4f}")
    print(f"Average LapJV error:          {np.mean(list(map_err_lapjv.values())):.4f}")
    print(f"Average flow inversion error: {np.mean(list(err_flow_composition.values())):.4f}")
    print(f"Average flow Hungarian error: {np.mean(list(err_flow_h_composition.values())):.4f}")
    print(f"Average flow LapJV error:     {np.mean(list(err_flow_lapjv_composition.values())):.4f}")


    plot_matching_error(err_flow_composition, map_err_knn, output_dir)

    with open(f'{output_dir}/err_flow_composition.csv', 'w+', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['source', 'target', 'error_flow', 'error_knn'])
        for (source, target) in err_flow_composition:
            err_flow = err_flow_composition[(source, target)]
            err_knn = map_err_knn[(source, target)]
            writer.writerow([source, target, err_flow, err_knn])


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


    args = parser.parse_args()
    if args.run_name == "":
        args.run_name = input("Enter a run name: ")

    print("------------------------------------------")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print("------------------------------------------")

    a = 1

    main(args)
