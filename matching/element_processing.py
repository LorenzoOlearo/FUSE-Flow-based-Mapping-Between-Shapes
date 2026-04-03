from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
import trimesh
from tqdm import tqdm

from matching.data_structures import DataPath, Element
from model.models import FMCond
from model.networks import MLP, GeomDist
from utils.dataset_utils import get_common_landmarks_between_two_models
from utils.mesh_utils import (
    compute_geodesic_distances_pointcloud,
    mesh_geodesics,
    pointcloud_geodesics,
)


def get_mesh_element_features(
    element: str,
    mesh: trimesh.Trimesh,
    data_path: DataPath,
    landmarks: List[int],
    recompute: bool,
    device: str,
) -> torch.Tensor:

    vertex_features_path = Path(
        data_path.flows_path,
        element,
        f"vertex-features-{data_path.features_type}-norm.npy",
    )
    vertex_features = torch.tensor(np.load(vertex_features_path).astype(np.float32)).to(
        device
    )

    features_path = Path(
        data_path.flows_path, element, f"features-{data_path.features_type}-norm.npy"
    )
    features = torch.tensor(np.load(features_path).astype(np.float32)).to(device)

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

    return features, vertex_features


def get_sdf_element_features(
    element: str,
    data_path: DataPath,
    device: str,
) -> torch.Tensor:

    # We load the features already normalized by the diameter as used to train the SDF flows
    features_path = Path(data_path.sdf_path, element, "features-landmarks-norm.npy")
    vertex_features_path = Path(
        data_path.sdf_path, element, "vertex-features-landmarks-norm.npy"
    )

    try:
        tqdm.write(
            f"Loading precomputed features for {element} from {data_path.features_path}"
        )
        features = torch.tensor(np.load(features_path).astype(np.float32)).to(device)
        vertex_features = torch.tensor(
            np.load(vertex_features_path).astype(np.float32)
        ).to(device)
    except Exception as e:
        raise ValueError(f"Error loading features from {vertex_features_path}: {e}")

    return features, vertex_features


def get_pt_element_features(
    element: str,
    data_path: DataPath,
    device: str,
    recompute: bool,
) -> torch.Tensor:

    if data_path.dataset == "KINECT":
        pt = trimesh.load(
            Path(data_path.dataset_path, element + data_path.dataset_extension),
            process=False,
        )
        pt = trimesh.Trimesh(vertices=pt.vertices, faces=[], process=False)
        features_path = Path(
            data_path.flows_path,
            element,
            f"vertex-features-{data_path.features_type}-norm.npy",
        )

        if recompute:
            tqdm.write(
                f"Computing features for {element} in {data_path.features_path} with heat method"
            )
            features = torch.tensor(
                compute_geodesic_distances_pointcloud(
                    mesh=pt, source_index=data_path.landmarks
                ),
                dtype=torch.float32,
            ).T.to(device)
        elif features_path.exists() and not recompute:
            tqdm.write(
                f"Loading precomputed features for {element} from {features_path}"
            )
            features = np.load(features_path).astype(np.float32)
            features = torch.tensor(features).to(device)
        else:
            raise ValueError(
                f"Features file {features_path} does not exist and recompute is set to False."
            )

    elif data_path.dataset == "FAUST":
        features_path = Path(
            data_path.scan_features_path,
            element,
            f"vertex-features-{data_path.features_type}-norm.npy",
        )
        tqdm.write(f"Loading precomputed features for {element} from {features_path}")
        features = np.load(features_path).astype(np.float32)

    return features


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
            corr = (
                np.loadtxt(data_path.corr_path / f"44_{element}.txt").astype(int)
                + data_path.corr_offset
            )
            landmarks = corr[faust_landmarks]
        else:
            corr = np.arange(len(mesh.vertices))
            landmarks = faust_landmarks

        if element == "39":
            corr = (
                np.loadtxt(data_path.corr_path / f"44_{element}.txt").astype(int)
                + data_path.corr_offset
            )
            landmarks = np.array([12467, 5360, 329, 4886, 375]) - 1
            tqdm.write(
                f"[SHREC19] Using custom landmarks for element {element}: {landmarks}"
            )
            return corr, landmarks
        elif element == "28":
            corr = (
                np.loadtxt(data_path.corr_path / f"44_{element}.txt").astype(int)
                + data_path.corr_offset
            )
            landmarks = np.array([7227, 42204, 51876, 32591, 48136]) - 1
            tqdm.write(
                f"[SHREC19] Using custom landmarks for element {element}: {landmarks}"
            )
            return corr, landmarks

    # Default case
    corr = _load_correspondence_file(
        element, data_path.corr_path, mesh, data_path.corr_offset
    )
    landmarks = corr[data_path.landmarks]
    return corr, landmarks


def _load_correspondence_file(element, corr_path, mesh, corr_offset):
    """Load correspondence indices from file if available."""
    if corr_path is None:
        tqdm.write("[CORR] No correspondence path provided, using identity mapping.")
        return np.arange(len(mesh.vertices))

    # TODO: special case for TOSCA
    if corr_path.name == "sym":
        element_name = "".join(filter(str.isalpha, element))
        path = Path(corr_path, element_name + ".sym.labels")
    else:
        path = Path(corr_path, element + ".vts")

    return np.loadtxt(path).astype(int) + corr_offset


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

    if model is not None:
        model_path = Path(data_path.flows_path, element, "checkpoint-best.pth")
        model.load_state_dict(
            torch.load(model_path, weights_only=False)["model"], strict=True
        )
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
    vertex_points = np.load(
        Path(data_path.sdf_path, element, "vertex-voxel-projection.npy")
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

    if model is not None:
        model_path = Path(data_path.flows_SDFs_path, element, "checkpoint-9999.pth")
        model.load_state_dict(
            torch.load(model_path, weights_only=False)["model"], strict=True
        )
        model.to(device).eval()

    return Element(
        element=element,
        representation="sdf",
        features=features,
        vertex_features=vertex_features,
        points=torch.as_tensor(points, dtype=torch.float32, device=device),
        vertex_points=torch.as_tensor(
            vertex_points, dtype=torch.float32, device=device
        ),
        model=model,
        mesh=mesh,
        landmarks=landmarks,
        corr=corr,
        dists=dists,
        diameter=diameter,
    )


def _process_pt_element(element, mesh, model, device, data_path) -> Element:
    """Process point cloud-based representation element."""
    if data_path.dataset == "KINECT":
        pt = trimesh.load(
            Path(data_path.dataset_path, element + data_path.dataset_extension),
            process=False,
        )
        pt = trimesh.Trimesh(vertices=pt.vertices, faces=[], process=False)
        corr = (
            np.loadtxt(Path(data_path.corr_path, element + ".vts")).astype(int)
            + data_path.corr_offset
        )
        landmarks = corr[data_path.landmarks]
        vertex_points = torch.as_tensor(pt.vertices, dtype=torch.float32, device=device)

        vertex_features = get_pt_element_features(
            element=element, data_path=data_path, device=device, recompute=False
        )

        dists, diameter = pointcloud_geodesics(
            pt=pt, target=element, recompute=False, dists_path=str(data_path.dists_path)
        )

        if model is not None:
            model_path = Path(data_path.flows_path, element, "checkpoint-9999.pth")
            model.load_state_dict(
                torch.load(model_path, weights_only=False)["model"], strict=True
            )
            model.to(device).eval()

        return Element(
            element=element,
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

    elif data_path.dataset == "FAUST":
        tqdm.write("Loading FAUST element from scan point cloud")
        element_scan = element.replace("tr_reg", "tr_scan")
        pt = trimesh.load(
            Path(
                data_path.scan_dataset_path,
                element_scan + data_path.scan_dataset_extension,
            ),
            process=False,
        )
        pt = trimesh.Trimesh(vertices=pt.vertices, faces=[], process=False)

        vertex_points = torch.as_tensor(pt.vertices, dtype=torch.float32, device=device)

        if data_path.scan_corr_path is None:
            tqdm.write(
                "[CORR] No correspondence path provided, using identity mapping."
            )
            corr = np.arange(len(mesh.vertices))
        else:
            corr = np.loadtxt(
                Path(data_path.scan_corr_path, element.replace("reg", "scan") + ".vts")
            ).astype(int)
        landmarks = data_path.landmarks

        vertex_features = torch.tensor(
            get_pt_element_features(
                element=element_scan,
                data_path=data_path,
                device=device,
                recompute=False,
            ),
            dtype=torch.float32,
            device=device,
        )

        vertex_features = vertex_features[corr]
        vertex_points = vertex_points[corr]
        corr = np.arange(len(vertex_features))

        # we take the diamter of the faust pt so that we can normalize the
        # geodesic features; we then evaluate the geodesic error on the mesh
        # dists matrix to be consistent with the SDF case
        _, diameter = pointcloud_geodesics(
            pt=pt,
            target=element_scan,
            recompute=False,
            dists_path=str(data_path.scan_dists_path),
        )

        dists, diameter_mesh = mesh_geodesics(
            mesh=mesh,
            target=element,
            recompute=False,
            dists_path=str(data_path.dists_path),
        )

        # Align the new dists over the scan diameter
        dists = dists * (diameter / diameter_mesh)

        if model is not None:
            model_path = Path(
                data_path.scan_flows_path, element_scan, "checkpoint-9999.pth"
            )
            model.load_state_dict(
                torch.load(model_path, weights_only=False)["model"], strict=True
            )
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


def get_network(
    network_selection: str,
    embedding_dim: int,
    mlp_hidden_size: int,
    mlp_depth: int,
    mlp_num_frequencies: int,
):
    if network_selection == "MLP":
        network = MLP(
            channels=embedding_dim,
            hidden_size=mlp_hidden_size,
            depth=mlp_depth,
            num_frequencies=mlp_num_frequencies,
        )
    elif network_selection == "GEOMDIST":
        network = GeomDist(channels=embedding_dim)
    else:
        raise ValueError(
            f"Unknown network '{network_selection}'. Choose 'MLP' or 'GEOMDIST'."
        )

    return network


def process_element(
    element: str,
    representation: str,
    device: str,
    mesh_baseline: bool,
    features_normalization: str,
    data_path: "DataPath",
    embedding_dim: int,
    mlp_hidden_size: int = 256,
    mlp_depth: int = 4,
    mlp_num_frequencies: int = -1,
    network_selection: str = "MLP",
    edm_preconditioning: bool = False,
    load_flow: bool = True,
):
    """
    Process a dataset element according to the chosen representation.
    Returns a fully populated Element object.

    If load_flow=False, no checkpoint is loaded and the model is set to None;
    Use this when running only non-FUSE methods
    """
    tqdm.write(f"Loading element '{element}' with representation '{representation}'")

    mesh_path = Path(data_path.dataset_path, element + data_path.dataset_extension)
    mesh = trimesh.load(mesh_path, process=False) if mesh_baseline else None

    if load_flow:
        model = FMCond(
            channels=embedding_dim,
            network=get_network(
                network_selection=network_selection,
                embedding_dim=embedding_dim,
                mlp_hidden_size=mlp_hidden_size,
                mlp_depth=mlp_depth,
                mlp_num_frequencies=mlp_num_frequencies,
            ),
            use_edm_preconditioning=edm_preconditioning,
        ).to(device)
    else:
        model = None

    if representation == "mesh":
        element = _process_mesh_element(element, mesh, model, device, data_path)
    elif representation == "sdf":
        element = _process_sdf_element(element, mesh, model, device, data_path)
    elif representation == "pt":
        element = _process_pt_element(element, mesh, model, device, data_path)
    else:
        raise ValueError(f"Invalid representation: '{representation}'")

    return element
