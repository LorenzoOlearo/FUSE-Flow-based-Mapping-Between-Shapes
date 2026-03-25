from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import trimesh


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
    features_type: str = "landmarks"
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
