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

SDFs_PATH = Path('./out/SDFs')
FAUST_PATH = Path('./data/MPI-FAUST/training/registrations/')
N_LANDMARKS = 5
FLOWS_PATH = Path('./out/flows')
MATCHING_PATH = Path('./out/matching')


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
    """Determine which targets to process based on CLI arguments."""
    targets = [
        f.name for f in SDFs_PATH.iterdir() if f.is_dir() and
        any(child.name.endswith('-features.txt') for child in f.iterdir()) and
        (FLOWS_PATH / f.name / 'checkpoint-9999.pth').is_file() and
        is_in_range(f.name)
    ]
    print(f"Processing all targets: {targets}")
    return targets


def process_element(element:str, device: str, mesh_baseline: bool):
    mesh_path = Path(FAUST_PATH, element + '.ply')
    mesh = trimesh.load(mesh_path, process=False)
    mesh = normalize_mesh(mesh)

    if mesh_baseline is False:
        features_path = Path(SDFs_PATH, element, f'{element}-sdf-dijkstra-features.txt')
        points_path = Path(SDFs_PATH, element, f'{element}-sdf-sampled-points.txt')
        points = torch.tensor(np.loadtxt(points_path).astype(np.float32)).to(device)
    else:
        features_path = Path(SDFs_PATH, element, f'{element}-sdf-mesh-dists.txt')
        points = torch.tensor(mesh.vertices.astype(np.float32)).to(device)

    features = torch.tensor(np.loadtxt(features_path).astype(np.float32)).to(device)

    model = FMCond(channels=N_LANDMARKS, network=MLP(channels=N_LANDMARKS).to(device))
    model.load_state_dict(torch.load(Path(FLOWS_PATH, element, 'checkpoint-9999.pth'), weights_only=False)['model'], strict=True)
    model.to(device)
    model.eval()

    return features, points, model


def process_pair(source: str, target: str, device, mesh_baseline: bool, plot_html: bool, plot_png: bool):
    source_features, source_points, source_model = process_element(source, device, mesh_baseline)
    target_features, target_points, target_model = process_element(target, device, mesh_baseline)

    p2p = compute_p2p_with_geomdist(source_features, target_features, source_model, target_model)
    matched_points = target_points[p2p]

    err = torch.norm(matched_points - target_points, dim=-1).mean().item()
    print(f"Flow inversion error {source} -> {target}: {err:.4f}")

    source_points = source_points[:100000]
    target_points = target_points[:100000]

    os.makedirs(MATCHING_PATH, exist_ok=True)
    start_end_subplot(
        source_points, matched_points,
        plots_path=str(MATCHING_PATH),
        run_name=f'matching{source}-{target}',
        show=False,
        html=plot_html,
        png=plot_png,
    )

    return err


def main(args):
    device = 'cuda:1'
    torch.cuda.set_device(device)
    mesh_baseline = args.mesh_baseline

    targets = get_targets()

    err_flow_inversion = {}
    for source in targets:
        for target in targets:
            if source == target:
                continue
            print(f"Processing {source} -> {target}")
            err = process_pair(source, target, device, mesh_baseline, args.plot_html, args.plot_png)
            err_flow_inversion[(source, target)] = err

    with open(f'{MATCHING_PATH}/err_flow_inversion.csv', 'w+', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['source', 'target', 'error'])
        for (source, target), err in err_flow_inversion.items():
            writer.writerow([source, target, err])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate SDF model on mesh")
    parser.add_argument('--mesh_baseline', action='store_true', help="Use the mesh vertices features instead of randomly sampled points", default=False)
    parser.add_argument('--plot_png', action='store_true', help="Save a PNG plot of the matching", default=False)
    parser.add_argument('--plot_html', action='store_true', help="Save an HTML plot of the matching", default=False)

    args = parser.parse_args()

    print("------------------------------------------")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print("------------------------------------------")

    main(args)
