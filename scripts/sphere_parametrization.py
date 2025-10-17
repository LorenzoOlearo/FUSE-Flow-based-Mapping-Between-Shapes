import os 
import sys
import json
from pathlib import Path
import numpy as np
import torch
import trimesh
import argparse
import plotly.graph_objects as go

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import *
from model import models, networks
from model.models import EDMPrecond, FMCond


GAUSSIAN_TO_SHAPE_MODEL_PATH = "out/sphere_parametrization/xyz_tr_reg_080/checkpoint-9999.pth"
GAUSSIAN_TO_SPHERE_MODEL_PATH = "out/sphere_parametrization/xyz_sphere/checkpoint-9999.pth"


def plot_uv_points(points, u, v):
    # Linear combination of u and v
    a, b = 0.5, 0.5
    c = a * u + b * v

    # Normalize to [0, 1] for coloring
    c_min, c_max = np.min(c), np.max(c)
    c_norm = (c - c_min) / (c_max - c_min + 1e-8)

    idx = slice(0, min(100_000, len(points)))
    fig = go.Figure(data=[go.Scatter3d(
        x=points[idx, 0],
        y=points[idx, 1],
        z=points[idx, 2],
        mode='markers',
        marker=dict(size=2, color=c_norm[idx], colorscale='Viridis', opacity=0.8, showscale=True)
    )])
    fig.update_layout(scene=dict(aspectmode='data'))
    fig.show()


def main(args):
    if args.config:
        with open(args.config, "r") as f:
            config_args = json.load(f)
        for key, value in config_args.items():
            if not hasattr(args, key):
                continue
            setattr(args, key, value)

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        setup_logging(args.output_dir)

    device = initialize_device_and_seed(args)

    target_mesh = trimesh.load(args.data_path, process=False)
    mesh_points, _= target_mesh.sample(500_000, return_index=True)
    mesh_points = torch.tensor(mesh_points, dtype=torch.float32).to(device)

    gaussian_to_shape_model = FMCond(
            channels=args.embedding_dim,
            depth=args.depth,
            network=networks.__dict__[args.network](channels=args.embedding_dim),
        )
    gaussian_to_shape_model.to(device)
    gaussian_to_shape_model.load_state_dict(
        torch.load(
            GAUSSIAN_TO_SHAPE_MODEL_PATH,
            map_location=device,
            weights_only=False,
        )["model"],
        strict=True,
    )

    pullback = gaussian_to_shape_model.inverse(samples=mesh_points, num_steps=64)

    gaussian_to_sphere_model = FMCond(
            channels=args.embedding_dim,
            depth=args.depth,
            network=networks.__dict__[args.network](channels=args.embedding_dim),
        )
    gaussian_to_sphere_model.to(device)
    gaussian_to_sphere_model.load_state_dict(
        torch.load(
            GAUSSIAN_TO_SPHERE_MODEL_PATH,
            map_location=device,
            weights_only=False,
        )["model"],
        strict=True,
    )

    reconstructed_sphere_points = gaussian_to_sphere_model.sample(noise=pullback, num_steps=64)

    # Parametrize sphere points with 2 values (u, v) in [0,1] using spherical coords
    r = torch.linalg.norm(reconstructed_sphere_points, dim=-1, keepdim=True).clamp_min(1e-8)
    p = reconstructed_sphere_points / r
    y, z, x = p.unbind(-1)
    theta = torch.atan2(y, x) # [-pi, pi]
    phi = torch.acos(z.clamp(-1 + 1e-6, 1 - 1e-6))  # [0, pi]
    u = (theta + np.pi) / (2 * np.pi)
    v = phi / np.pi
    
    u = u.cpu().numpy()
    v = v.cpu().numpy()
    plot_uv_points(reconstructed_sphere_points.cpu().numpy(), u, v)
    plot_uv_points(mesh_points.cpu().numpy(), u, v)


if __name__ == "__main__":
    args = get_inline_arg()

    main(args)
