import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import trimesh
from geomfum.metric import HeatDistanceMetric
from geomfum.shape.mesh import TriangleMesh
from geomfum.shape.point_cloud import PointCloud
from tqdm import tqdm

from matching.data_structures import DataPath, Element, MatchingResult
from util.dataset_utils import get_common_landmarks_between_two_models
from util.metrics import (
    compute_coverage,
    compute_dirichlet_energy,
    compute_geodesic_error,
)


def approx_max_euclidean_distance(points: torch.Tensor, sample_size: int) -> float:
    """Approximate the maximum Euclidean distance using random sampling."""
    num_points = points.shape[0]
    if num_points <= sample_size:
        return torch.cdist(points, points).max().item()

    idx1 = torch.randperm(num_points)[:sample_size]
    sampled_points = points[idx1]
    cd = torch.cdist(sampled_points, sampled_points)
    return cd.max().item()


def get_geodesic_dists(
    mesh: trimesh.Trimesh, element: str, data_path: DataPath
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


# ---------------------------------------------------------------------------
# Per-dataset metric evaluation
# Each function returns (matched_points, euclidean_error, geodesic_error,
#                        dirichlet_energy, coverage).
# matched_points is returned because some branches reassign it.
# ---------------------------------------------------------------------------


def evaluate_shrec20(
    p2p: np.ndarray,
    matched_points: torch.Tensor,
    target_points: torch.Tensor,
    source_element: Element,
    target_element: Element,
    target_element_dists: np.ndarray,
    gts_path: str,
    max_euclidean_error: float,
) -> Tuple:
    source_gt_path = Path(gts_path) / f"{source_element.element}.mat"
    target_gt_path = Path(gts_path) / f"{target_element.element}.mat"

    _, common_source_landmarks, common_target_landmarks = (
        get_common_landmarks_between_two_models(source_gt_path, target_gt_path)
    )

    # Indices of the descriptor landmarks within the common landmark sets
    source_landmark_indices = [
        np.where(common_source_landmarks == lmk)[0][0]
        for lmk in source_element.landmarks
        if lmk in common_source_landmarks
    ]
    target_landmark_indices = [
        np.where(common_target_landmarks == lmk)[0][0]
        for lmk in target_element.landmarks
        if lmk in common_target_landmarks
    ]

    matched_points_subset = matched_points[common_source_landmarks]
    target_subset = target_points[common_target_landmarks]

    # Remove the descriptor landmarks from the evaluation
    matched_points_subset = torch.tensor(
        np.delete(matched_points_subset.cpu().numpy(), source_landmark_indices, axis=0)
    ).to(matched_points.device)
    target_subset = torch.tensor(
        np.delete(target_subset.cpu().numpy(), target_landmark_indices, axis=0)
    ).to(target_subset.device)

    euclidean_error = (
        torch.norm(matched_points_subset - target_subset, dim=-1).mean().item()
        / max_euclidean_error
    )
    geodesic_error = compute_geodesic_error(
        target_element_dists, p2p, common_source_landmarks, common_target_landmarks
    )
    coverage = compute_coverage(
        p2p, source_element.vertex_points, target_element.vertex_points
    )
    dirichlet_energy = compute_dirichlet_energy(
        source_element.mesh, target_element.mesh, p2p
    ).item()

    tqdm.write(
        f"[SHREC20] Evaluated on {len(common_source_landmarks)} landmarks "
        f"between {source_element.element} and {target_element.element}"
    )
    return matched_points, euclidean_error, geodesic_error, dirichlet_energy, coverage


def evaluate_shrec19(
    p2p: np.ndarray,
    source_element: Element,
    target_element: Element,
    gts_path: str,
    data_path: DataPath,
    max_euclidean_error: float,
) -> Tuple:
    gt_path = Path(gts_path) / f"{source_element.element}_{target_element.element}.txt"
    gt = np.loadtxt(gt_path).astype(int) + data_path.corr_offset

    matched_points = target_element.vertex_points[p2p]
    target_points_corr = target_element.vertex_points[gt]

    euclidean_error = (
        torch.norm(matched_points - target_points_corr, dim=-1).mean().item()
        / max_euclidean_error
    )
    geodesic_error = compute_geodesic_error(
        target_element.dists, p2p, np.arange(len(gt)), gt
    )
    dirichlet_energy = compute_dirichlet_energy(
        source_element.mesh, target_element.mesh, p2p
    ).item()
    coverage = compute_coverage(
        p2p, source_element.vertex_points, target_element.vertex_points
    )
    return matched_points, euclidean_error, geodesic_error, dirichlet_energy, coverage


def evaluate_faust_scan(
    p2p: np.ndarray,
    source_element: Element,
    target_element: Element,
    data_path: DataPath,
    max_euclidean_error: float,
) -> Tuple:
    matched_points = target_element.vertex_points[p2p]
    target_points = target_element.vertex_points

    euclidean_error = (
        torch.norm(matched_points - target_points, dim=-1).mean().item()
        / max_euclidean_error
    )
    geodesic_error = compute_geodesic_error(
        target_element.dists, p2p, source_element.corr, target_element.corr
    )

    # Dirichlet energy is evaluated on the original meshes, not the point clouds
    source_mesh = trimesh.load(
        Path(
            data_path.dataset_path,
            source_element.element.replace("scan", "reg")
            + data_path.scan_dataset_extension,
        ),
        process=False,
    )
    target_mesh = trimesh.load(
        Path(
            data_path.dataset_path,
            target_element.element.replace("scan", "reg")
            + data_path.scan_dataset_extension,
        ),
        process=False,
    )
    dirichlet_energy = compute_dirichlet_energy(source_mesh, target_mesh, p2p).item()
    coverage = compute_coverage(
        p2p, source_element.vertex_points, target_element.vertex_points
    )
    return matched_points, euclidean_error, geodesic_error, dirichlet_energy, coverage


def evaluate_default(
    p2p: np.ndarray,
    matched_points: torch.Tensor,
    target_points_corr: torch.Tensor,
    source_element: Element,
    target_element: Element,
    max_euclidean_error: float,
) -> Tuple:
    euclidean_error = (
        torch.norm(matched_points - target_points_corr, dim=-1).mean().item()
        / max_euclidean_error
    )
    geodesic_error = compute_geodesic_error(
        target_element.dists, p2p, source_element.corr, target_element.corr
    )
    dirichlet_energy = compute_dirichlet_energy(
        source_element.mesh, target_element.mesh, p2p
    ).item()
    coverage = compute_coverage(
        p2p, source_element.vertex_points, target_element.vertex_points
    )
    return matched_points, euclidean_error, geodesic_error, dirichlet_energy, coverage


# ---------------------------------------------------------------------------


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
            max_euclidean_error = approx_max_euclidean_distance(
                target_points, sample_size=50_000
            )

        if gts_path is not None and "shrec20" in str(gts_path).lower():
            (
                matched_points,
                euclidean_error,
                geodesic_error,
                dirichlet_energy,
                coverage,
            ) = evaluate_shrec20(
                p2p,
                matched_points,
                target_points,
                source_element,
                target_element,
                target_element_dists,
                gts_path,
                max_euclidean_error,
            )
        elif gts_path is not None and "shrec19" in str(gts_path).lower():
            (
                matched_points,
                euclidean_error,
                geodesic_error,
                dirichlet_energy,
                coverage,
            ) = evaluate_shrec19(
                p2p,
                source_element,
                target_element,
                gts_path,
                data_path,
                max_euclidean_error,
            )
        elif data_path.dataset == "FAUST" and (
            source_element.representation == "pt"
            or target_element.representation == "pt"
        ):
            (
                matched_points,
                euclidean_error,
                geodesic_error,
                dirichlet_energy,
                coverage,
            ) = evaluate_faust_scan(
                p2p,
                source_element,
                target_element,
                data_path,
                max_euclidean_error,
            )
        else:
            (
                matched_points,
                euclidean_error,
                geodesic_error,
                dirichlet_energy,
                coverage,
            ) = evaluate_default(
                p2p,
                matched_points,
                target_points_corr,
                source_element,
                target_element,
                max_euclidean_error,
            )

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
        np.save(
            p2p_dir
            / f"p2p-{name}-{source_element.element}-{target_element.element}.npy",
            p2p,
        )
        tqdm.write(f"Saved P2P mapping for {name} at {p2p_dir}")

    return results


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
        return (
            name,
            run_matching_methods(
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
            )[name],
        )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(wrapper, name, fn): name
            for name, fn in matching_methods.items()
        }
        for future in as_completed(futures):
            name, res = future.result()
            results[name] = res

    return results


def log_results(source: str, target: str, results: Dict[str, MatchingResult]) -> None:
    """tqdm.write error metrics nicely formatted."""
    tqdm.write(f"> Evaluation results for {source} -> {target}:")
    for name, res in results.items():
        tqdm.write(
            f"  > {name:<20}: euclidean_error {res.euclidean_error:.4f} | geodesic_error {res.geodesic_error:.4f} | dirichlet_energy={res.dirichlet_energy:.4f} | coverage={res.coverage:.4f} | elapsed={res.elapsed:.2f}s"
        )
