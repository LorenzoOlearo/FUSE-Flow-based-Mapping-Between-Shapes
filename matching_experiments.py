import os
os.environ["GEOMSTATS_BACKEND"] = "pytorch"
from pathlib import Path
from typing import List
import argparse
import csv
import numpy as np
import torch
import trimesh
import matplotlib.colors as mcolors
import time
from typing import Callable, Dict, Optional, Tuple
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
from geomfum.shape.point_cloud import PointCloud
from geomfum.metric import HeatDistanceMetric
import potpourri3d as pp3d

from util.matching_utils import *
from util.metrics import *
from util.dataset_utils import *
from util.plot import plot_points, start_end_subplot
from util.mesh_utils import (
    mesh_geodesics,
    pointcloud_geodesics,
    compute_geodesic_distances_pointcloud
)


@dataclass
class DataPath:
    dataset: str
    output_dir: Path
    landmarks: List[int]
    dataset_path: Path
    dists_path: Path
    features_path: Path
    dataset_extension: str
    flows_path: Path
    flows_SDFs_path: Path | None
    sdf_path: Path | None
    corr_path: Optional[Path] = None
    corr_offset: int = 0
    
    # Attributes for SCAN-FAUST dataset when using point cloud representation on FAUST
    scan_flows_path: Path | None = None
    scan_dataset_path: Path | None = None
    scan_dataset_extension: str | None = None
    scan_dists_path: Path | None = None
    scan_features_path: Path | None = None
    scan_corr_path: Optional[Path] = None

    # Common landmarks CSV for SHREC20, see dataset_utils.py on how to get this file
    common_landmarks_path: Optional[Path] = None
    
    # Path to ground truth correspondences, needed because we evaluate the shrec20 dataset on the landmarks common to each pair of matched shapes
    gts_path: Optional[Path] = None


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
        if args.source_rep == "sdf" or args.target_rep == "sdf" or args.source_rep == "pt" or args.target_rep == "pt":
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

def get_targets_kinect(flows_path) -> List[str]:
    """Determine which targets to process."""
    targets = [
        f.name
        for f in flows_path.iterdir()
        if f.is_dir()
        and f.name.startswith(("data"))
    ]

    tqdm.write(f"Processing all targets: {targets}")
    return targets


def get_targets_surreal(flows_path) -> List[str]:
    targets = []
    targets = [
        f.name
        for f in flows_path.iterdir()
        if f.is_dir()
        and f.name.startswith(("surreal"))
    ]
    return targets


def get_targets_smplx(flows_path) -> List[str]:
    targets = []
    targets = [
        f.name
        for f in flows_path.iterdir()
        if f.is_dir()
        and f.name.startswith(("SMPLX"))
    ]
    return targets


def get_targets_shrec20(flows_path) -> List[str]:
    """Determine which targets to process."""
    targets = [
        f.name
        for f in flows_path.iterdir()
        if f.is_dir()
        and (flows_path / f.name / 'checkpoint-9999.pth').is_file()
    ]

    tqdm.write(f"Processing targets: {targets}")
    return targets


def get_targets_shrec19(flows_path) -> List[str]:
    """Determine which targets to process."""
    targets = [
        f.name
        for f in flows_path.iterdir()
        if f.is_dir()
        and (flows_path / f.name / 'checkpoint-9999.pth').is_file()
    ]

    tqdm.write(f"Processing targets: {targets}")
    return targets


def get_mesh_element_features(
    element: str,
    mesh: trimesh.Trimesh,
    data_path: DataPath,
    landmarks: List[int],
    recompute: bool,
    device: str,
) -> torch.Tensor:

    vertex_features_path = Path(data_path.flows_path, element, f"vertex-geodesics-vnorm.txt")
    vertex_features = torch.tensor(np.loadtxt(vertex_features_path).astype(np.float32)).to(device)
    
    features_path = Path(data_path.flows_path, element, f"vertex-geodesics-interpolated-vnorm.txt")
    features = torch.tensor(np.loadtxt(features_path).astype(np.float32)).to(device)
    
    tqdm.write("------------------------------------")
    tqdm.write(f"loaded vertex_features (shape {list(vertex_features.shape)}):")
    tqdm.write(f"  min: {vertex_features.min(dim=0).values.tolist()}")
    tqdm.write(f"  max: {vertex_features.max(dim=0).values.tolist()}")
    tqdm.write(f"  avg: {vertex_features.mean(dim=0).tolist()}")
    tqdm.write("------------------------------------")
    tqdm.write(f"loaded features (shape {list(vertex_features.shape)}):")
    tqdm.write(f"  min: {features.min(dim=0).values.tolist()}")
    tqdm.write(f"  max: {features.max(dim=0).values.tolist()}")
    tqdm.write(f"  avg: {features.mean(dim=0).tolist()}")

    # if recompute:
    #     tqdm.write(f"Computing features for {element} in {data_path.features_path}")
    #     vertex_features = torch.tensor(compute_geodesic_distances(mesh, landmarks).T.astype(np.float32)).to(device)
    #
    #     # DEBUG: WE SKIP THIS AS WE LOAD THE NORMALIZED FEATURES USED TO TRAIN THE FLOWS
    #     # geo_dists = get_geodesic_dists(mesh, element, data_path)
    #     # vertex_features /= geo_dists.max()
    #
    #     os.makedirs(vertex_features_path.parent, exist_ok=True)
    #     np.savetxt(vertex_features_path, vertex_features.cpu().numpy())
    #

    # if vertex_features_path.exists() and not recompute:
    #     tqdm.write(f"Loading precomputed features for {element} from {data_path.features_path}")
    #     vertex_features = torch.tensor(np.loadtxt(vertex_features_path).astype(np.float32)).to(device)
    # else:
    #     tqdm.write(f"Computing features for {element} in {data_path.features_path}")
    #     vertex_features = torch.tensor(compute_geodesic_distances(mesh, landmarks).T.astype(np.float32)).to(device)
    #
    #     # DEBUG: WE SKIP THIS AS WE LOAD THE NORMALIZED FEATURES USED TO TRAIN THE FLOWS
    #     # geo_dists = get_geodesic_dists(mesh, element, data_path)
    #     # vertex_features /= geo_dists.max()
    #
    #     os.makedirs(vertex_features_path.parent, exist_ok=True)
    #     np.savetxt(vertex_features_path, vertex_features.cpu().numpy())

    return features, vertex_features


def get_sdf_element_features(
    element: str,
    data_path: DataPath,
    device: str,
) -> torch.Tensor:

    # We load the featues already normalized by the diameter as used to train the SDF flows
    features_path = Path(data_path.sdf_path, element, f"{element}-geodesics-normalized-diameter.txt")
    vertex_features_path = Path(data_path.sdf_path, element, f"{element}-vertex-geodesics-normalized-diameter.txt")

    try:
        tqdm.write(f"Loading precomputed features for {element} from {data_path.features_path}")
        features = torch.tensor(np.loadtxt(features_path).astype(np.float32)).to(device)
        vertex_features = torch.tensor(np.loadtxt(vertex_features_path).astype(np.float32)).to(device)
    except Exception as e:
        raise ValueError(f"Error loading features from {vertex_features_path}: {e}")

    return features, vertex_features


def get_pt_element_features(
    element: str,
    data_path: DataPath,
    device: str,
    recompute: bool,
) -> torch.Tensor:

    pt = trimesh.load(
        Path(data_path.scan_dataset_path, element + data_path.dataset_extension),
        process=False
    )

    if data_path.dataset == "KINECT":
        features_path = Path(data_path.flows_path, element, f"vertex-geodesics-vnorm.txt")

        if recompute:
            tqdm.write(f"Computing features for {element} in {data_path.features_path} with heat method")
            features = torch.tensor(
                compute_geodesic_distances_pointcloud(
                    mesh=pt,
                    source_index=data_path.landmarks
                )
            ).T.to(device)
        elif features_path.exists() and not recompute:
            tqdm.write(f"Loading precomputed features for {element} from {features_path}")
            features = np.loadtxt(features_path).astype(np.float32)
            features = torch.tensor(features).to(device)
        else:
            raise ValueError(f"Features file {features_path} does not exist and recompute is set to False.")
        
    elif data_path.dataset == "FAUST":
        features_path = Path(data_path.scan_features_path, element, f"vertex-geodesics-vnorm.txt")
        tqdm.write(f"Loading precomputed features for {element} from {features_path}")
        features = np.loadtxt(features_path).astype(np.float32)

    return features


@dataclass
class Element:
    element: str
    representation: str
    features: torch.Tensor
    vertex_features: torch.Tensor
    points: torch.Tensor
    vertex_points: torch.Tensor
    model: torch.nn.Module
    mesh: trimesh.Trimesh
    landmarks: np.ndarray
    corr: np.ndarray | None
    dists: np.ndarray | None
    diameter: float | None


def process_element(
    element: str,
    representation: str,
    device: str,
    mesh_baseline: bool,
    features_normalization: str,
    data_path: "DataPath",
):
    """
    Process a dataset element according to the chosen representation.
    Returns a fully populated Element object.
    """
    tqdm.write(f"Loading element '{element}' with representation '{representation}'")

    mesh_path = Path(data_path.dataset_path, element + data_path.dataset_extension)
    mesh = trimesh.load(mesh_path, process=False) if mesh_baseline else None

    model = _init_model(len(data_path.landmarks), device)

    if representation == "mesh":
        element = _process_mesh_element(element, mesh, model, device, data_path)
    elif representation == "sdf":
        element = _process_sdf_element(element, mesh, model, device, data_path)
    elif representation == "pt":
        element = _process_pt_element(element, mesh, model, device, data_path)
    else:
        raise ValueError(f"Invalid representation: '{representation}'")

    return element


def _init_model(num_channels: int, device: str):
    """Initialize the functional map conditioning model."""
    return FMCond(
        channels=num_channels,
        network=MLP(channels=num_channels).to(device),
    )


def _process_mesh_element(element, mesh, model, device, data_path) -> Element:
    """Process a mesh-based representation element."""
    corr, landmarks = _load_landmarks_and_correspondences(element, mesh, data_path)

    points, _ = trimesh.sample.sample_surface(mesh, 100_000)
    points = torch.as_tensor(points, dtype=torch.float32, device=device)
    vertex_points = torch.as_tensor(mesh.vertices, dtype=torch.float32, device=device)

    features, vertex_features = get_mesh_element_features(
        element=element,
        mesh=mesh,
        data_path=data_path,
        landmarks=data_path.landmarks,
        device=device,
        recompute=False,
    )

    dists, diameter = mesh_geodesics(
        mesh=mesh,
        target=element,
        recompute=False,
        dists_path=str(data_path.dists_path),
    )

    model_path = Path(data_path.flows_path, element, "checkpoint-9999.pth")
    model.load_state_dict(torch.load(model_path, weights_only=False)["model"], strict=True)
    model.to(device).eval()

    return Element(
        element=element,
        representation="mesh",
        features=features,
        vertex_features=vertex_features,
        points=points,
        vertex_points=vertex_points,
        model=model,
        mesh=mesh,
        landmarks=landmarks,
        corr=corr,
        dists=dists,
        diameter=diameter,
    )


def _process_sdf_element(element, mesh, model, device, data_path) -> Element:
    """Process an SDF-based representation element."""
    vertex_points = np.loadtxt(
        Path(data_path.sdf_path, element, f"{element}-vertex-voxel-projection.txt")
    ).astype(np.float32)

    points = np.loadtxt(
        Path(data_path.sdf_path, element, f"{element}-sdf-sampled-surface-points.txt")
    ).astype(np.float32)

    features, vertex_features = get_sdf_element_features(
        element=element, data_path=data_path, device=device
    )

    corr = np.arange(len(vertex_points))
    landmarks = np.array(data_path.landmarks)

    dists, diameter = mesh_geodesics(
        mesh=mesh, target=element, recompute=False, dists_path=str(data_path.dists_path)
    )

    model_path = Path(data_path.flows_SDFs_path, element, "checkpoint-9999.pth")
    model.load_state_dict(torch.load(model_path, weights_only=False)["model"], strict=True)
    model.to(device).eval()

    return Element(
        element=element,
        representation="sdf",
        features=features,
        vertex_features=vertex_features,
        points=torch.as_tensor(points, dtype=torch.float32, device=device),
        vertex_points=torch.as_tensor(vertex_points, dtype=torch.float32, device=device),
        model=model,
        mesh=mesh,
        landmarks=landmarks,
        corr=corr,
        dists=dists,
        diameter=diameter,
    )


def _process_pt_element(element, mesh, model, device, data_path) -> Element:
    """Process a point cloud-based representation element."""
    # Check if the dataset is kinect
    if data_path.dataset == "KINECT":
        # corr = _load_correspondence_file(element, data_path.corr_path, mesh, data_path.corr_offset)
        if data_path.corr_path is None:
            tqdm.write("[CORR] No correspondence path provided, using identity mapping.")
            return np.arange(len(mesh.vertices))
        else:
            corr = np.loadtxt(Path(data_path.corr_path, element + ".vts")).astype(int) + data_path.corr_offset
            
        landmarks = corr[data_path.landmarks]
        

        features = get_pt_element_features(
            element=element, data_path=data_path, device=device, recompute=False
        )

        dists, diameter = pointcloud_geodesics(
            pt=mesh, target=element, recompute=False, dists_path=str(data_path.dists_path)
        )

        model_path = Path(data_path.flows_path, element, "checkpoint-9999.pth")
        model.load_state_dict(torch.load(model_path, weights_only=False)["model"], strict=True)
        model.to(device).eval()

        return Element(
            element=element,
            representation="pt",
            features=features,
            vertex_features=features,
            points=None,
            vertex_points=torch.as_tensor(mesh.vertices, dtype=torch.float32, device=device),
            model=model,
            mesh=mesh,
            landmarks=landmarks,
            corr=corr,
            dists=dists,
            diameter=diameter,
        )

    elif data_path.dataset == "FAUST":
        tqdm.write("Loading FAUST element from 10k point cloud")
        element_scan = element.replace("tr_reg", "tr_scan")
        pt = trimesh.load(
            Path(data_path.scan_dataset_path, element_scan + data_path.scan_dataset_extension),
            process=False
        )
        pt = trimesh.Trimesh(vertices=pt.vertices, faces=[], process=False)
        
        vertex_points = torch.as_tensor(pt.vertices, dtype=torch.float32, device=device)
       
        if data_path.scan_corr_path is None:
            tqdm.write("[CORR] No correspondence path provided, using identity mapping.")
            corr = np.arange(len(mesh.vertices))
        else:
            corr = np.loadtxt(Path(data_path.scan_corr_path, element.replace("reg", "scan") + ".vts")).astype(int)
        landmarks = data_path.landmarks
        
        vertex_features = torch.tensor(get_pt_element_features(
            element=element_scan, data_path=data_path, device=device, recompute=False
        ), dtype=torch.float32, device=device)
       
        vertex_features = vertex_features[corr] 
        vertex_points = vertex_points[corr]
        corr = np.arange(len(vertex_features))
        
        dists, diameter = pointcloud_geodesics(
            pt=pt, target=element_scan, recompute=False, dists_path=str(data_path.scan_dists_path)
        )
        
        model_path = Path(data_path.scan_flows_path, element_scan, "checkpoint-9999.pth")
        model.load_state_dict(torch.load(model_path, weights_only=False)["model"], strict=True)
        model.to(device).eval()
        
        return Element(
            element=element_scan,
            representation="pt",
            features=None,
            vertex_features=vertex_features,
            points=None,
            vertex_points=vertex_points,
            model=model,
            mesh=pt,
            landmarks=landmarks,
            corr=corr,
            dists=dists,
            diameter=diameter,
        )


def _load_landmarks_and_correspondences(element, mesh, data_path):
    """Handle SHREC19/20 dataset-specific landmark logic and default correspondences."""
    # SHREC20 fix
    if (
        data_path.landmarks == [-1, -1, -1, -1, -1, -1]
        and data_path.common_landmarks_path is not None
    ):
        df = pd.read_csv(data_path.common_landmarks_path)
        row = df[df["Model"] == f"{element}.obj"]
        if not row.empty:
            landmarks = row.iloc[0, 1:].values.astype(int).tolist()
            tqdm.write(f"[SHREC20] Common landmarks loaded for {element}: {landmarks}")
            corr = np.arange(len(mesh.vertices))
            return corr, landmarks

    # SHREC19 fix
    if "shrec19" in str(data_path.dataset_path).lower():
        tqdm.write("[SHREC19] Using SHREC19 correspondences from GT files")
        faust_landmarks = np.array([412, 5891, 6593, 3323, 2119])
        if element != "44":
            corr = np.loadtxt(data_path.corr_path / f"44_{element}.txt").astype(int) + data_path.corr_offset
            landmarks = corr[faust_landmarks]
        else:
            corr = np.arange(len(mesh.vertices))
            landmarks = faust_landmarks
        return corr, landmarks

    # Default case
    corr = _load_correspondence_file(element, data_path.corr_path, mesh, data_path.corr_offset)
    landmarks = corr[data_path.landmarks]
    return corr, landmarks


def _load_correspondence_file(element, corr_path, mesh, corr_offset):
    """Load correspondence indices from file if available."""
    if corr_path is None:
        tqdm.write("[CORR] No correspondence path provided, using identity mapping.")
        return np.arange(len(mesh.vertices))
    path = Path(corr_path, element + ".vts")
    return np.loadtxt(path).astype(int) + corr_offset


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
    source_sdf_projected_vertex_points=None,
    target_sdf_projected_vertex_points=None,
):
    """Return mapping of strategy names to their compute functions."""

    if matching_methods == "fast":
        return {
            "knn": lambda: compute_p2p_with_knn(source_features, target_features),
            "flow": lambda: compute_p2p_with_flows_composition(
                source_features, target_features, source_model, target_model, device
            ),
        }
    elif matching_methods == "sdf":
        return {
            "knn": lambda: compute_p2p_with_knn(source_features, target_features),
            "ot": lambda: compute_p2p_with_ot(source_features, target_features),
            "flow": lambda: compute_p2p_with_flows_composition(
                source_features, target_features, source_model, target_model, device
            ),
            "ndp-sdf": lambda: compute_p2p_with_ndp_sdf(
                source_vertex=source_sdf_projected_vertex_points,
                target_vertex=target_sdf_projected_vertex_points,
                source_landmarks=source_landmarks,
                target_landmarks=target_landmarks,
            ),
            "knn-in-gauss": lambda: compute_p2p_with_inverted_flows_in_gauss(
                source_features, target_features, source_model, target_model
            ),
        }
    elif matching_methods == "all":
        return {
            "knn": lambda: compute_p2p_with_knn(source_features, target_features),
            "ot": lambda: compute_p2p_with_ot(source_features, target_features),
            "fmaps": lambda: compute_p2p_with_fmaps(
                source_path, target_path, source_features, target_features
            ),
            "fmap-zoomout": lambda: compute_p2p_with_fmap_zoomout(
                source_path, target_path, source_features, target_features
            ),
            "fmap-neural-zoomout": lambda: compute_p2p_with_fmap_neural_zoomout(
                source_path, target_path, source_features, target_features
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
            "flow": lambda: compute_p2p_with_flows_composition(
                source_features, target_features, source_model, target_model, device
            ),
            "knn-in-gauss": lambda: compute_p2p_with_inverted_flows_in_gauss(
                source_features, target_features, source_model, target_model
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
            "fmap": lambda: compute_p2p_with_fmaps(
                source_path, target_path, source_features, target_features
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


def run_matching_methods_parallel(
    matching_methods,
    target_points,
    source_element,
    target_element,
    source_mesh,
    target_mesh,
    dists,
    output_dir: Path,
    gt_path: Optional[str] = None,
    source_corr=None,
    target_corr=None,
    max_workers=10,
):
    results = {}

    def wrapper(name, fn):
        return name, run_matching_methods(
            matching_methods={name: fn},
            target_points=target_points,
            source_element=source_element,
            target_element=target_element,
            source_mesh=source_mesh,
            target_mesh=target_mesh,
            dists=dists,
            output_dir=output_dir,
            gts_path=gt_path,
            source_corr=source_corr,
            target_corr=target_corr,
        )[name]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(wrapper, name, fn): name for name, fn in matching_methods.items()}
        for future in as_completed(futures):
            name, res = future.result()
            results[name] = res

    return results


def approx_max_euclidean_distance(points: torch.Tensor, sample_size: int) -> float:
    """Approximate the maximum Euclidean distance using random sampling."""
    num_points = points.shape[0]
    if num_points <= sample_size:
        return torch.cdist(points, points).max().item()

    idx1 = torch.randperm(num_points)[:sample_size]
    sampled_points = points[idx1]
    cd = torch.cdist(sampled_points, sampled_points)
    max_dist = cd.max().item()
    return max_dist


def run_matching_methods(
    matching_methods: dict[str, callable],
    target_points: torch.Tensor,
    source_element: Element,
    target_element: Element,
    target_element_dists: np.ndarray,
    output_dir: Path,
    data_path: DataPath,
    gts_path: str | None = None,
) -> dict[str, MatchingResult]:
    """
    Run all P2P strategies and compute matched points + errors.
    Assumes Elements are already permuted and aligned (corr applied in process_element).
    """
    results = {}

    for name, func in matching_methods.items():
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        p2p, elapsed = func()
        
        matched_points = target_points[p2p[source_element.corr]]
        target_points_corr = target_element.vertex_points[target_element.corr]

        if target_points.shape[0] <= 50_000:
            max_euclidean_error = torch.cdist(target_points, target_points).max().item()
        else:
            max_euclidean_error = approx_max_euclidean_distance(target_points, sample_size=50_000)

        # SHREC20
        if gts_path is not None and "shrec20" in str(gts_path).lower():
            source_gt_path = Path(gts_path) / f"{source_element.element}.mat"
            target_gt_path = Path(gts_path) / f"{target_element.element}.mat"
            if source_gt_path.exists() and target_gt_path.exists():
                _, common_source_landmarks, common_target_landmarks = get_common_landmarks_between_two_models(
                    source_gt_path, target_gt_path
                )
                matched_points_subset = matched_points[common_source_landmarks]
                target_subset = target_points[common_target_landmarks]
                euclidean_error = torch.norm(matched_points_subset - target_subset, dim=-1).mean().item() / max_euclidean_error
                geodesic_error = compute_geodesic_error(target_element_dists, p2p, common_source_landmarks, common_target_landmarks) / target_element_dists.max()
                coverage = compute_coverage(p2p[common_source_landmarks], common_source_landmarks, common_target_landmarks)
                dirichlet_energy = compute_dirichlet_energy(source_element.mesh, target_element.mesh, p2p).item()
                tqdm.write(f"[SHREC20] Evaluated on {len(common_source_landmarks)} landmarks between {source_element.element} and {target_element.element}")
        
        # SHREC19
        elif gts_path is not None and "shrec19" in str(gts_path).lower():
            gt_path = Path(gts_path) / f"{source_element.element}_{target_element.element}.txt"
            gt = np.loadtxt(gt_path).astype(int) + data_path.corr_offset
            matched_points = target_element.vertex_points[p2p]
            target_points_corr = target_element.vertex_points[gt]
            euclidean_error = torch.norm(matched_points - target_points_corr, dim=-1).mean().item() / max_euclidean_error
            geodesic_error = compute_geodesic_error(target_element.dists, p2p, np.arange(len(gt)), gt)
            dirichlet_energy = compute_dirichlet_energy(source_element.mesh, target_element.mesh, p2p).item()
            coverage = compute_coverage(p2p, source_element.vertex_points, target_element.vertex_points)
        
        # FAUST SCAN
        elif data_path.dataset == "FAUST" and (source_element.representation == "pt" or target_element.representation == "pt"):
            matched_points = target_element.vertex_points[p2p]
            target_points = target_element.vertex_points
            euclidean_error = torch.norm(matched_points - target_points, dim=-1).mean().item() / max_euclidean_error
            # geodesic_error = compute_geodesic_error(target_element.dists, p2p, source_element.corr, target_element.corr)
            geodesic_error = - 1
           
            # we evaluate the dirichlet energy of the p2p on the meshes
            source_mesh = trimesh.load(
                Path(data_path.dataset_path, source_element.element.replace("scan", "reg") + data_path.scan_dataset_extension),
                process=False
            )
            target_mesh = trimesh.load(
                Path(data_path.dataset_path, target_element.element.replace("scan", "reg") + data_path.scan_dataset_extension),
                process=False
            )
            dirichlet_energy = compute_dirichlet_energy(source_mesh, target_mesh, p2p).item()
            coverage = compute_coverage(p2p, source_element.vertex_points, target_element.vertex_points)
            
        else:
            euclidean_error = torch.norm(matched_points - target_points_corr, dim=-1).mean().item() / max_euclidean_error
            geodesic_error = compute_geodesic_error(target_element.dists, p2p, source_element.corr, target_element.corr)
            dirichlet_energy = compute_dirichlet_energy(source_element.mesh, target_element.mesh, p2p).item()
            coverage = compute_coverage(p2p, source_element.vertex_points, target_element.vertex_points)

        results[name] = MatchingResult(
            indices=p2p,
            matched_points=matched_points,
            euclidean_error=euclidean_error,
            geodesic_error=geodesic_error,
            dirichlet_energy=dirichlet_energy,
            coverage=coverage,
            elapsed=elapsed,
        )

        p2p_dir = output_dir / "p2p"
        p2p_dir.mkdir(parents=True, exist_ok=True)
        np.save(p2p_dir / f"p2p-{name}-{source_element.element}-{target_element.element}.npy", p2p)
        tqdm.write(f"Saved P2P mapping for {name} at {p2p_dir}")

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
    target_points: torch.Tensor,
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
            target_points[res.indices[:max_points]],
            plots_path=str(output_dir),
            run_name=f"{name}-{source}-{target}",
            show=False,
            html=plot_html,
            png=plot_png,
        )


def get_geodesic_dists(
    mesh: trimesh.Trimesh,
    element: str,
    data_path: DataPath
) -> np.ndarray:
    """
    Compute or load geodesic distance matrix for a element mesh.

    Args:
        mesh (trimesh.Trimesh): The mesh to compute distances on.
        element_name (str): Identifier for caching (e.g. filename stem).
        data_path (DataPath): Dataclass containing paths for caching.
    Returns:
        np.ndarray: Geodesic distance matrix of shape (n_vertices, n_vertices).
    """
    os.makedirs(data_path.dists_path, exist_ok=True)
    dist_cache_path = os.path.join(data_path.dists_path, f"{element}_dists.npy")

    if os.path.exists(dist_cache_path):
        dist = np.load(dist_cache_path)
    else:
        if len(mesh.faces) > 0:
            mesh_gf = TriangleMesh(mesh.vertices, np.array(mesh.faces))
            heat = HeatDistanceMetric.from_registry(mesh_gf)
            dist = heat.dist_matrix()
        else:
            pc_gf = PointCloud(np.array(mesh.vertices))
            heat = HeatDistanceMetric.from_registry(mesh=False, shape=pc_gf)
            dist = heat.dist_matrix()
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
    data_path: "DataPath",
    output_dir: str,
) -> pd.DataFrame:
    """
    Main pipeline to process a source-target pair for shape matching.
    Handles loading, feature extraction, matching, evaluation, and visualization.

    Args:
        source, target (str): Source and target element identifiers.
        source_rep, target_rep (str): Representation types ('mesh', 'sdf', or 'pt').
        device (str): Device to use for computation.
        mesh_baseline (bool): Whether to use the mesh as the base geometry.
        plot_html, plot_png (bool): Whether to export visualization plots.
        all_methods (str): Matching methods to evaluate.
        features_normalization (str): Feature normalization scheme.
        data_path (DataPath): Container for dataset paths.
        output_dir (str): Directory to store results.

    Returns:
        pd.DataFrame: Results table containing evaluation metrics for all methods.
    """

    source_element = process_element(
        element=source,
        representation=source_rep,
        device=device,
        mesh_baseline=mesh_baseline,
        features_normalization=features_normalization,
        data_path=data_path,
    )

    target_element = process_element(
        element=target,
        representation=target_rep,
        device=device,
        mesh_baseline=mesh_baseline,
        features_normalization=features_normalization,
        data_path=data_path,
    )

    matching_methods = get_matching_methods(
        source_features=source_element.vertex_features,
        target_features=target_element.vertex_features,
        source_path=Path(data_path.dataset_path, source + data_path.dataset_extension),
        target_path=Path(data_path.dataset_path, target + data_path.dataset_extension),
        source_model=source_element.model,
        target_model=target_element.model,
        source_landmarks=source_element.landmarks,
        target_landmarks=target_element.landmarks,
        device=device,
        matching_methods=all_methods,
        source_sdf_projected_vertex_points=source_element.vertex_points,
        target_sdf_projected_vertex_points=target_element.vertex_points,
    )

    results = run_matching_methods(
        matching_methods=matching_methods,
        target_points=target_element.vertex_points,
        source_element=source_element,
        target_element=target_element,
        target_element_dists=target_element.dists,
        output_dir=output_dir,
        gts_path=data_path.gts_path,
        data_path=data_path,
    )

    log_results(source, target, results)

    plot_results(
        source_points=source_element.vertex_points,
        target_points=target_element.vertex_points,
        source=source,
        target=target,
        results=results,
        output_dir=str(data_path.output_dir),
        plot_html=plot_html,
        plot_png=plot_png,
        max_points=100_000,
    )

    df = pd.DataFrame(
        [
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
            for name, res in results.items()
        ]
    )

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
        if args.source_rep == "pt" or args.target_rep == "pt":
            raise ValueError("The 'pt' representation is only supported for the KINECT dataset.")
        dataset = "SMAL"
        targets = get_targets_smal(Path(config["matching_config"][dataset]["flows_path"]))
    elif args.surreal:
        if args.source_rep == "pt" or args.target_rep == "pt":
            raise ValueError("The 'pt' representation is only supported for the KINECT dataset.")
        dataset = "SURREAL"
        targets = get_targets_surreal(Path(config["matching_config"][dataset]["flows_path"]))
    elif args.kinect:
        # if args.pt_skinning or args.source_rep != "pt" or args.target_rep != "pt":
        #     raise ValueError("For KINECT dataset only 'pt' representation is supported.")
        dataset = "KINECT"
        targets = get_targets_kinect(Path(config["matching_config"][dataset]["flows_path"]))
    elif args.smplx:
        dataset = "SMPLX"
        targets = get_targets_smplx(Path(config["matching_config"][dataset]["flows_path"]))
    elif args.shrec20:
        dataset = "SHREC20"
        targets = get_targets_shrec20(Path(config["matching_config"][dataset]["flows_path"]))
    elif args.shrec19:
        dataset = "SHREC19"
        targets = get_targets_shrec19(Path(config["matching_config"][dataset]["flows_path"]))
    else:
        raise ValueError("Please specify either --faust, --smal, --kinect, --surreal, --smplx, --shrec20, or --shrec19 to select the dataset.")

    if targets is None or len(targets) == 0:
        raise ValueError("No targets found to process.")
    tqdm.write(f"Running {dataset} experiments")
    tqdm.write(f"Found {len(targets)} targets: {targets}")

    output_dir = Path(f"./out/matching/{dataset.lower()}/matching-{args.source_rep}-{args.target_rep}-{args.matching_run_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    tqdm.write(f"Output directory: {output_dir}")

    data_path = DataPath(
        dataset=dataset,
        output_dir=output_dir,
        landmarks=config["matching_config"][dataset]["landmarks"],
        dataset_path=Path(config["matching_config"][dataset]["dataset_path"]),
        dists_path=Path(config["matching_config"][dataset]["dists_path"]),
        features_path=Path(config["matching_config"][dataset]["features_path"]),
        dataset_extension=config["matching_config"][dataset]["dataset_extension"],
        flows_path=Path(config["matching_config"][dataset]["flows_path"]),
        flows_SDFs_path=Path(config["matching_config"][dataset]["flows_SDFs_path"]) if "flows_SDFs_path" in config["matching_config"][dataset] else None,
        sdf_path=Path(config["matching_config"][dataset]["SDFs_path"]) if "SDFs_path" in config["matching_config"][dataset] else None,
        corr_path=Path(config["matching_config"][dataset]["corr_path"]) if "corr_path" in config["matching_config"][dataset] else None,
        corr_offset=config["matching_config"][dataset]["corr_offset"] if "corr_offset" in config["matching_config"][dataset] else 0,
        common_landmarks_path=Path(config["matching_config"][dataset]["common_landmarks_path"]) if dataset == "SHREC20" else None,
        gts_path=Path(config["matching_config"][dataset]["gts_path"]) if "gts_path" in config["matching_config"][dataset] else None,
        scan_flows_path=Path(config["matching_config"][dataset]["scan_flows_path"]) if "scan_flows_path" in config["matching_config"][dataset] else None,
        scan_dataset_path=Path(config["matching_config"][dataset]["scan_dataset_path"]) if "scan_dataset_path" in config["matching_config"][dataset] else None,
        scan_dists_path=Path(config["matching_config"][dataset]["scan_dists_path"]) if "scan_dists_path" in config["matching_config"][dataset] else None,
        scan_features_path=Path(config["matching_config"][dataset]["scan_features_path"]) if "scan_features_path" in config["matching_config"][dataset] else None,
        scan_dataset_extension=config["matching_config"][dataset]["scan_dataset_extension"] if "scan_dataset_extension" in config["matching_config"][dataset] else None,
        scan_corr_path=Path(config["matching_config"][dataset]["scan_corr_path"]) if "scan_corr_path" in config["matching_config"][dataset] else None,
    )
        

    results = []
    times = []
    results_file = Path(output_dir, "matching_results.csv")

    if args.same:
        pairs = [(t, t) for t in targets]

    elif args.pt_skinning and dataset == "KINECT":
        sources = get_targets_smplx(Path(config["matching_config"]["SMPLX"]["smplx_template_flows_path"]))
        targets = get_targets_kinect(Path(config["matching_config"][dataset]["flows_path"]))
        pairs = [(s, t) for s in sources for t in targets]
    elif dataset == "SHREC19":
        # Only load the pairs defined in the corr_path
        if data_path.corr_path is None:
            raise ValueError("For SHREC19 dataset, corr_path must be provided in the config to define the matching pairs.")
        corr_files = [f for f in os.listdir(data_path.corr_path) if f.endswith(".txt")]
        pairs = []
        for f in corr_files:
            base = os.path.splitext(f)[0]
            parts = base.split('_')
            if len(parts) != 2:
                tqdm.write(f"Skipping file with unexpected format (expected 'target_source.txt'): {f}")
            source, target = parts[0], parts[1]
            if source in targets and target in targets:
                if source != "43" and target != "43":
                    pairs.append((source, target))
    else:
        # Default case in which we match all different pairs
        pairs = [(s, t) for s in targets for t in targets if s != t]

    # Override pairs if specific source and target shapes are provided
    if args.source_shape is not None and args.target_shape is not None:
        if args.source_shape not in targets:
            raise ValueError(f"Source shape '{args.source_shape}' not found in the targets list.")
        if args.target_shape not in targets:
            raise ValueError(f"Target shape '{args.target_shape}' not found in the targets list.")
        pairs = [(args.source_shape, args.target_shape)]
        
    tqdm.write(f"Processing {len(pairs)} shape pairs.")
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
            output_dir=output_dir,
        )
        elapsed_time = time.perf_counter() - start_time
        tqdm.write(f"Time taken for {source} -> {target}: {elapsed_time:.2f} seconds")
        times.append(elapsed_time)

        df.to_csv(results_file, mode="a", header=not results_file.exists(), index=False)
        results.append(df)

    results_df = pd.concat(results, ignore_index=True)
    results_df.to_csv(Path(output_dir, "matching_results_completed.csv"), index=False)
    
    # Save and tqdm.write average time
    times_sec = [t.total_seconds() if hasattr(t, "total_seconds") else float(t) for t in times]
    avg_time = sum(times_sec) / len(times_sec)
    tqdm.write(f"Average time for all pairs: {avg_time:.2f} seconds")
    with open(Path(output_dir, "timing_results.txt"), "w") as f:
        for i, t in enumerate(times_sec):
            f.write(f"Pair {i + 1}: {t:.2f} seconds\n")
        f.write(f"\nAverage time: {avg_time:.2f} seconds\n")

    tqdm.write("Average errors across all pairs:")
    avg_metrics = results_df.groupby("method")[
        ["euclidean_error", "geodesic_error", "dirichlet", "coverage", "elapsed"]
    ].mean()
    avg_metrics.to_csv(Path(output_dir, "matching_results_average.csv"))
    tqdm.write(avg_metrics.to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Matching experiments for SDFs and meshes"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:1",
        help="Device to use (e.g., 'cpu', 'cuda:1, 'cuda:1')",
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
        "--surreal",
        action="store_true",
        help="Run matching methods on SURREAL dataset",
        default=False,
    )
    parser.add_argument(
        "--kinect",
        action="store_true",
        help="Run matching methods on KINECT dataset",
        default=False,
    )
    parser.add_argument(
        "--smplx",
        action="store_true",
        help="Run matching methods on the SMPLX template meshes",
        default=False,
    )
    parser.add_argument(
        "--shrec20",
        action="store_true",
        help="Run matching methods on the SHREC20 dataset",
        default=False,
    )
    parser.add_argument(
        "--shrec19",
        action="store_true",
        help="Run matching methods on the SHREC19 dataset",
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
        help="Representation of the first element in the pair ('mesh', 'sdf', or 'pt')",
    )
    parser.add_argument(
        "--target_rep",
        type=str,
        default="mesh",
        help="Representation of the first element in the pair ('mesh', 'sdf', or 'pt')",
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
    parser.add_argument(
        "--correspondence_offset",
        default = 0,
        type = int,
        help="offset to apply to the correspondences, in case of SMAL offset : -1, otherwise default=0"
    )
    parser.add_argument(
        "--matching_run_name",
        type=str,
        default="",
        help="Name for the matching run, used for the output directory",
    )
    parser.add_argument(
        "--pt_skinning",
        action="store_true",
        default=False,
        help="Perform the SMPLX template skinning experiment with the Kinect point clouds",
    )

    parser.add_argument(
        "--source_shape",
        type=str,
        default=None,
        help="Filename of the source shape in the specified dataset. Overrides the default behavior of matching all pairs.",
    )

    parser.add_argument(
        "--target_shape",
        type=str,
        default=None,
        help="Filename of the target shape in the specified dataset. Overrides the default behavior of matching all pairs.",
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
