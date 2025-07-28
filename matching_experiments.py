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
    compute_p2p_with_geomdist,
    compute_p2p_with_knn_gauss,
    compute_p2p_with_knn,
    compute_p2p_with_fmaps
)
from util.mesh_utils import normalize_mesh
from model.models import FMCond
from model.networks import MLP
from util.plot import start_end_subplot
import matplotlib.pyplot as plt

SDFs_PATH = Path('./out/SDFs')
FAUST_PATH = Path('./data/MPI-FAUST/training/registrations/')
N_LANDMARKS = 5
FLOWS_PATH = Path('./out/flows_vertex')
MATCHING_PATH = Path('./out/matching_SDF-SDF-vertex-SAME')


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
        if i == 83:
            continue
        target = f"tr_reg_{i:03d}"
        targets.append(target)

    print(f"Processing all targets: {targets}")
    return targets


def process_element(element: str, representation: str, device: str, mesh_baseline: bool):
    mesh_path = Path(FAUST_PATH, element + '.ply')
    mesh = trimesh.load(mesh_path, process=False)
    mesh = normalize_mesh(mesh)

    if representation == 'mesh':
        features_path = Path(FLOWS_PATH, element, f'features.txt')
        points = torch.tensor(mesh.vertices.astype(np.float32)).to(device)
        features = torch.tensor(np.loadtxt(features_path).astype(np.float32)).to(device)
    elif representation == 'sdf':
        print(f"Processing {element} with SDF representation")
        if mesh_baseline is False:
            features_path = Path(SDFs_PATH, element, f'{element}-sdf-dijkstra-features.txt')
            points_path = Path(SDFs_PATH, element, f'{element}-sdf-sampled-points.txt')
            points = torch.tensor(np.loadtxt(points_path).astype(np.float32)).to(device)
        else:
            print(f"Using mesh baseline for {element}")
            features_path = Path(SDFs_PATH, element, f'{element}-sdf-mesh-dists.txt')
            points = torch.tensor(mesh.vertices.astype(np.float32)).to(device)
        features = torch.tensor(np.loadtxt(features_path).astype(np.float32)).to(device)

    model = FMCond(channels=N_LANDMARKS, network=MLP(channels=N_LANDMARKS).to(device))
    model.load_state_dict(torch.load(Path(FLOWS_PATH, element, 'checkpoint-9999.pth'), weights_only=False)['model'], strict=True)
    model.to(device)
    model.eval()

    return features, points, model


def process_pair(source: str, target: str, source_rep: str, target_rep: str, device: str, mesh_baseline: bool, plot_html: bool, plot_png: bool):
    source_features, source_points, source_model = process_element(source, source_rep, device, mesh_baseline)
    target_features, target_points, target_model = process_element(target, target_rep, device, mesh_baseline)

    p2p_flow = compute_p2p_with_geomdist(source_features, target_features, source_model, target_model)
    p2p_knn = compute_p2p_with_knn(source_features, target_features)
    matched_points_flow = target_points[p2p_flow]
    matched_points_knn = target_points[p2p_knn]

    err_flow = torch.norm(matched_points_flow - target_points, dim=-1).mean().item()
    err_knn = torch.norm(target_points[p2p_knn] - target_points, dim=-1).mean().item()
    print(f"> Flow inversion error: {source} -> {target}: {err_flow:.4f}")
    print(f"> KNN error:            {source} -> {target}: {err_knn:.4f}")

    source_points = source_points[:100000]
    target_points = target_points[:100000]

    os.makedirs(MATCHING_PATH, exist_ok=True)
    start_end_subplot(
        source_points, matched_points_flow,
        plots_path=str(MATCHING_PATH),
        run_name=f'flow-{source}-{target}',
        show=False,
        html=plot_html,
        png=plot_png,
    )

    os.makedirs(MATCHING_PATH, exist_ok=True)
    start_end_subplot(
        source_points, matched_points_knn,
        plots_path=str(MATCHING_PATH),
        run_name=f'knn-{source}-{target}',
        show=False,
        html=plot_html,
        png=plot_png,
    )

    return err_flow, err_knn


def plot_matching_error(err_flow_inversion, map_err_knn):
    shapes = sorted(set([k[0] for k in err_flow_inversion.keys()] + [k[1] for k in err_flow_inversion.keys()]))
    plt.figure(figsize=(14, 8))
    color_map = plt.get_cmap('tab20')
    for idx, shape in enumerate(shapes):
        color = color_map(idx % 20)
        x = []
        y_flow = []
        y_knn = []
        for (source, target), err_flow in err_flow_inversion.items():
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
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.xticks(rotation=45)
    plt.tight_layout(rect=[0, 0, 0.75, 1])
    plt.savefig(f'{MATCHING_PATH}/plot_errors_all_shapes.png', bbox_inches='tight')
    plt.close()


def main(args):
    device = 'cuda:1'
    torch.cuda.set_device(device)
    mesh_baseline = args.mesh_baseline

    targets = get_targets()
    err_flow_inversion = {}
    map_err_knn = {}

    if args.same:
        for target in targets:
            print(f"Matching the same shape {target} from {args.source_rep} to {args.target_rep}")
            err_flow, err_knn = process_pair(target, target, args.source_rep, args.target_rep, device, mesh_baseline, args.plot_html, args.plot_png)
            err_flow_inversion[(target, target)] = err_flow
            map_err_knn[(target, target)] = err_knn
    else:
        for source in targets:
            for target in targets:
                if source ~= target:
                    print(f"Processing {source} -> {target}")
                    err_flow, err_knn = process_pair(source, target, args.source_rep, args.target_rep, device, mesh_baseline, args.plot_html, args.plot_png)
                    err_flow_inversion[(source, target)] = err_flow
                    map_err_knn[(source, target)] = err_knn

    print(f"Average flow inversion error: {np.mean(list(err_flow_inversion.values())):.4f}")
    print(f"Average KNN error:            {np.mean(list(map_err_knn.values())):.4f}")

    plot_matching_error(err_flow_inversion, map_err_knn) 

    with open(f'{MATCHING_PATH}/err_flow_inversion.csv', 'w+', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['source', 'target', 'error_flow', 'error_knn'])
        for (source, target) in err_flow_inversion:
            err_flow = err_flow_inversion[(source, target)]
            err_knn = map_err_knn[(source, target)]
            writer.writerow([source, target, err_flow, err_knn])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate SDF model on mesh")
    parser.add_argument('--mesh_baseline', action='store_true', help="Use the mesh vertices features instead of randomly sampled points", default=False)
    parser.add_argument('--plot_png', action='store_true', help="Save a PNG plot of the matching", default=False)
    parser.add_argument('--plot_html', action='store_true', help="Save an HTML plot of the matching", default=False)
    parser.add_argument('--source_rep', type=str, default='mesh', help="Representation of the first element in the pair (e.g., 'mesh', 'sdf')")
    parser.add_argument('--target_rep', type=str, default='mesh', help="Representation of the first element in the pair (e.g., 'mesh', 'sdf')")
    parser.add_argument('--same', action='store_true', help="Match the same shape with different representations", default=False)

    args = parser.parse_args()

    print("------------------------------------------")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print("------------------------------------------")

    main(args)
