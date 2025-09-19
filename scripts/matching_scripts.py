import os
import numpy as np
import torch
import trimesh

import argparse
import json
from itertools import combinations
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.models import FMCond
from model.networks import MLP
from util.matching_utils import *
from util.mesh_utils import normalize_mesh_08 as normalize_mesh
from util.mesh_utils import pc_normalize
from util.metrics import (
    compute_geodesic_error,
    compute_dirichlet_energy,
    compute_coverage,
)
from geomfum.shape.mesh import TriangleMesh
from geomfum.metric.mesh import HeatDistanceMetric
import potpourri3d as pp3d


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run shape matching experiments with configuration"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the training configuration JSON file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device specified in config (e.g., 'cuda:0', 'cpu')",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="all",
        help="Comma-separated list of methods to run: geomdist,knn,knn_gauss,fmaps,ot,flows_composition_zoomout,knn_neural_zoomout,fmap_neural_zoomout,ndp_landmarks,fmap_zoomout,knn_zoomout,fmaps_wks",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="all",
        help="Comma-separated list of metrics to calculate: euclidean,geodesic,dirichlet,coverage",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override output directory specified in config",
    )
    return parser.parse_args()


def run_matching_experiments(config, args):
    # Extract dataset information from config structure
    if "dataset" in config:
        # This is a training config format
        dataset_config = config["dataset"]

        # Extract path information
        if dataset_config.get("type") == "directory":
            in_dir = dataset_config.get("directory", "./data/kinect/off_clean/")
        else:
            in_dir = dataset_config.get("data_path", "./data/")

        corr_dir = dataset_config.get(
            "correspondence_dir", "./data/kinect/corres_clean/"
        )

        file_extension = dataset_config.get("file_extension", ".off")
        corr_extension = dataset_config.get("correspondence_extension", ".vts")

        # Check for landmark offset (needed for some datasets like SMAL)
        landmark_offset = dataset_config.get("landmark_offset", 0)

        landmarks = dataset_config.get("landmarks", None)
        # Extract output and checkpoint directory from output section
        base_dir = config.get("output", {}).get("base_dir", "./experiments/results/")
        out_dir = (
            args.output_dir
            if args.output_dir
            else os.path.join(base_dir, "matching_results")
        )
        checkpoint_dir = base_dir

        # Get shapes based on dataset type
        if dataset_config.get("type") == "range":
            start = dataset_config.get("range_start", 0)
            end = dataset_config.get("range_end", 1)
            file_format = dataset_config.get("file_format", "{:d}")
            file_names = [file_format.format(i) for i in range(start, end)]
        else:
            # Directory type
            file_names = [
                f.split(".")[0]
                for f in os.listdir(in_dir)
                if f.endswith(file_extension)
            ]
    else:
        # Fallback to direct config parameters (older format)
        in_dir = config.get("data_dir", "./data/kinect/off_clean/")
        out_dir = (
            args.output_dir
            if args.output_dir
            else config.get("output_dir", "./experiments/results/")
        )
        landmarks = dataset_config.get("landmarks", None)

        corr_dir = config.get("correspondence_dir", "./data/kinect/corres_clean/")
        checkpoint_dir = config.get("checkpoint_dir", "./experiments/")
        file_extension = config.get("file_extension", ".off")
        corr_extension = config.get("correspondence_extension", ".vts")
        landmark_offset = config.get("landmark_offset", 0)

        # Get all file names in the directory
        if "shapes" in config:
            file_names = config["shapes"]
        else:
            file_names = [
                f.split(".")[0]
                for f in os.listdir(in_dir)
                if f.endswith(file_extension)
            ]

    # Setup device
    device = (
        args.device
        if args.device
        else config.get("training", {}).get("device", "cuda:0")
    )
    torch.cuda.set_device(device)

    # Ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)

    # Parse methods to run
    methods_arg = args.methods.lower()
    methods = {
        "geomdist": methods_arg == "all" or "geomdist" in methods_arg,
        "knn": methods_arg == "all" or "knn" in methods_arg,
        "knn_gauss": methods_arg == "all" or "knn_gauss" in methods_arg,
        "fmaps": methods_arg == "all" or "fmaps" in methods_arg,
        "ot": methods_arg == "all" or "ot" in methods_arg,
        "flows_composition_zoomout": methods_arg == "all"
        or "flows_composition_zoomout" in methods_arg,
        "knn_neural_zoomout": methods_arg == "all"
        or "knn_neural_zoomout" in methods_arg,
        "fmap_neural_zoomout": methods_arg == "all"
        or "fmap_neural_zoomout" in methods_arg,
        "ndp_landmarks": methods_arg == "all" or "ndp_landmarks" in methods_arg,
        "fmap_zoomout": methods_arg == "all" or "fmap_zoomout" in methods_arg,
        "knn_zoomout": methods_arg == "all" or "knn_zoomout" in methods_arg,
        "fmaps_wks": methods_arg == "all" or "fmaps_wks" in methods_arg,
    }

    # Parse metrics to calculate
    metrics_arg = args.metrics.lower()
    metrics = {
        "euclidean": metrics_arg == "all" or "euclidean" in metrics_arg,
        "geodesic": metrics_arg == "all" or "geodesic" in metrics_arg,
        "dirichlet": metrics_arg == "all" or "dirichlet" in metrics_arg,
        "coverage": metrics_arg == "all" or "coverage" in metrics_arg,
    }

    print(f"Running experiments on {len(file_names)} shapes using device: {device}")
    print(f"Methods enabled: {[m for m, enabled in methods.items() if enabled]}")
    print(f"Metrics enabled: {[m for m, enabled in metrics.items() if enabled]}")
    print(f"Data directory: {in_dir}")
    print(f"Checkpoint directory: {checkpoint_dir}")
    print(f"Output directory: {out_dir}")

    # Initialize result matrices for each method and metric
    results = {}

    # For each method, initialize result matrices for each metric
    for method_name in [m for m, enabled in methods.items() if enabled]:
        results[method_name] = {}
        for metric_name in [m for m, enabled in metrics.items() if enabled]:
            results[method_name][metric_name] = np.zeros(
                (len(file_names), len(file_names))
            )

    # Iterate over combinations of file names
    for source, target in combinations(file_names, 2):
        print(f"\nProcessing pair: {source} -> {target}")

        # Load meshes
        mesh1_path = os.path.join(in_dir, f"{source}{file_extension}")
        mesh2_path = os.path.join(in_dir, f"{target}{file_extension}")

        if not os.path.exists(mesh1_path) or not os.path.exists(mesh2_path):
            print("Warning: Mesh files not found. Skipping pair.")
            continue

        mesh1 = trimesh.load(mesh1_path, process=False)
        mesh2 = trimesh.load(mesh2_path, process=False)
        if len(mesh1.faces) > 0:
            mesh1 = normalize_mesh(trimesh.load(mesh1_path, process=False))
            mesh2 = normalize_mesh(trimesh.load(mesh2_path, process=False))
        else:
            mesh1.vertices = pc_normalize(mesh1.vertices)
            mesh2.vertices = pc_normalize(mesh2.vertices)

        # Load correspondences
        corr_a_path = os.path.join(corr_dir, f"{source}{corr_extension}")
        corr_b_path = os.path.join(corr_dir, f"{target}{corr_extension}")

        if not os.path.exists(corr_a_path) or not os.path.exists(corr_b_path):
            print("Warning: Correspondence files not found. Using Identity.")
            corr_a = np.arange(len(mesh1.vertices))
            corr_b = np.arange(len(mesh2.vertices))
        else:
            corr_a = np.array(np.loadtxt(corr_a_path, dtype=np.int32))
            corr_b = np.array(np.loadtxt(corr_b_path, dtype=np.int32))

            # Apply offset if needed (for datasets like SMAL that are 1-indexed)
            if landmark_offset != 0 or corr_dir == "../datasets/meshes/SMAL_r/corres/":
                corr_a = corr_a + landmark_offset
                corr_b = corr_b + landmark_offset

        # Target vertices for evaluation
        v2 = torch.tensor(mesh2.vertices).float().to(device)
        if len(mesh2.faces) > 0:
            # Try to load the distance matrix from cache if it exists
            dist_cache_dir = os.path.join(out_dir, "../../dist_cache")
            os.makedirs(dist_cache_dir, exist_ok=True)
            dist_cache_path = os.path.join(dist_cache_dir, f"{target}_dist.npy")
            if os.path.exists(dist_cache_path):
                dist = np.load(dist_cache_path)
            else:
                mesh_gf = TriangleMesh(np.array(mesh2.vertices), np.array(mesh2.faces))
                heat = HeatDistanceMetric.from_registry(mesh_gf)
                dist = heat.dist_matrix()
                np.save(dist_cache_path, dist)
        else:
            # Try to load the distance matrix from cache if it exists
            dist_cache_dir = os.path.join(out_dir, "../../dist_cache")
            os.makedirs(dist_cache_dir, exist_ok=True)
            dist_cache_path = os.path.join(dist_cache_dir, f"{target}_dist.npy")
            if os.path.exists(dist_cache_path):
                dist = np.load(dist_cache_path)
            else:
                solver = pp3d.PointCloudHeatSolver(np.array(mesh2.vertices))
                distances = []
                for idx in range(len(mesh2.vertices)):
                    distances.append(solver.compute_distance(idx))
                dist = np.array(distances)
                np.save(dist_cache_path, dist)

        # Load features
        feature_file_extension = config.get("feature_file_extension", ".txt")
        source_feature_path = os.path.join(
            checkpoint_dir, source, f"features{feature_file_extension}"
        )
        target_feature_path = os.path.join(
            checkpoint_dir, target, f"features{feature_file_extension}"
        )

        if not os.path.exists(source_feature_path) or not os.path.exists(
            target_feature_path
        ):
            print("Warning: Feature files not found. Skipping pair.")
            continue

        source_features = torch.tensor(
            np.loadtxt(source_feature_path).astype(np.float32)
        ).to(device)
        target_features = torch.tensor(
            np.loadtxt(target_feature_path).astype(np.float32)
        ).to(device)

        # Load models
        model_channels = config.get("training", {}).get("embedding_dim", 5)
        checkpoint_filename = config.get("execution", {}).get(
            "checkpoint_name", "checkpoint-9999.pth"
        )

        source_checkpoint_path = os.path.join(
            checkpoint_dir, source, checkpoint_filename
        )
        target_checkpoint_path = os.path.join(
            checkpoint_dir, target, checkpoint_filename
        )

        if not os.path.exists(source_checkpoint_path) or not os.path.exists(
            target_checkpoint_path
        ):
            print("Warning: Checkpoint files not found. Skipping pair.")
            continue

        source_model = FMCond(
            channels=model_channels,
            network=MLP(channels=model_channels).to(device),
        )
        source_model.to(device)
        source_model.load_state_dict(
            torch.load(source_checkpoint_path, map_location=device, weights_only=False)[
                "model"
            ],
            strict=True,
        )

        target_model = FMCond(
            channels=model_channels,
            network=MLP(channels=model_channels).to(device),
        )
        target_model.to(device)
        target_model.load_state_dict(
            torch.load(target_checkpoint_path, map_location=device, weights_only=False)[
                "model"
            ],
            strict=True,
        )

        source_model.eval()
        target_model.eval()

        source_idx = file_names.index(source)
        target_idx = file_names.index(target)

        source_landmarks = corr_a[landmarks]
        target_landmarks = corr_b[landmarks]
        # Compute correspondences with different methods and evaluate with different metrics

        # 1. Geometric distance (Flow Composition)
        if methods["geomdist"]:
            p2p = compute_p2p_with_flows_composition(
                source_features, target_features, source_model, target_model
            )

            # Evaluate with different metrics
            if metrics["euclidean"]:
                eucl_error = torch.mean(
                    torch.norm(v2.cpu()[p2p][corr_a] - v2.cpu()[corr_b], dim=-1)
                ).item()
                results["geomdist"]["euclidean"][source_idx, target_idx] = eucl_error
                print(f"Geomdist FLOW COMPOSITION - Euclidean: {eucl_error:.6f}")

            if metrics["geodesic"]:
                geo_error = compute_geodesic_error(dist, p2p, corr_a, corr_b)
                results["geomdist"]["geodesic"][source_idx, target_idx] = geo_error
                print(f"Geomdist FLOW COMPOSITION - Geodesic: {geo_error:.6f}")

            if metrics["dirichlet"]:
                dirichlet = compute_dirichlet_energy(mesh1, mesh2, p2p)
                results["geomdist"]["dirichlet"][source_idx, target_idx] = dirichlet
                print(f"Geomdist FLOW COMPOSITION - Dirichlet: {dirichlet:.6f}")

            if metrics["coverage"]:
                cov = compute_coverage(p2p, len(mesh2.vertices))
                results["geomdist"]["coverage"][source_idx, target_idx] = cov
                print(f"Geomdist FLOW COMPOSITION - Coverage: {cov:.6f}")

        # 2. KNN in Gaussian space
        if methods["knn_gauss"]:
            p2p = compute_p2p_with_inverted_flows_in_gauss(
                source_features, target_features, source_model, target_model
            )

            # Evaluate with different metrics
            if metrics["euclidean"]:
                eucl_error = torch.mean(
                    torch.norm(v2.cpu()[p2p][corr_a] - v2.cpu()[corr_b], dim=-1)
                ).item()
                results["knn_gauss"]["euclidean"][source_idx, target_idx] = eucl_error
                print(f"KNN IN GAUSSIAN - Euclidean: {eucl_error:.6f}")

            if metrics["geodesic"]:
                geo_error = compute_geodesic_error(dist, p2p, corr_a, corr_b)
                results["knn_gauss"]["geodesic"][source_idx, target_idx] = geo_error
                print(f"KNN IN GAUSSIAN - Geodesic: {geo_error:.6f}")

            if metrics["dirichlet"]:
                dirichlet = compute_dirichlet_energy(mesh1, mesh2, p2p)
                results["knn_gauss"]["dirichlet"][source_idx, target_idx] = dirichlet
                print(f"KNN IN GAUSSIAN - Dirichlet: {dirichlet:.6f}")

            if metrics["coverage"]:
                cov = compute_coverage(p2p, len(mesh2.vertices))
                results["knn_gauss"]["coverage"][source_idx, target_idx] = cov
                print(f"KNN IN GAUSSIAN - Coverage: {cov:.6f}")

        # 3. Plain KNN
        if methods["knn"]:
            p2p = compute_p2p_with_knn(source_features, target_features)

            # Evaluate with different metrics
            if metrics["euclidean"]:
                eucl_error = torch.mean(
                    torch.norm(v2.cpu()[p2p][corr_a] - v2.cpu()[corr_b], dim=-1)
                ).item()
                results["knn"]["euclidean"][source_idx, target_idx] = eucl_error
                print(f"KNN - Euclidean: {eucl_error:.6f}")

            if metrics["geodesic"]:
                geo_error = compute_geodesic_error(dist, p2p, corr_a, corr_b)
                results["knn"]["geodesic"][source_idx, target_idx] = geo_error
                print(f"KNN - Geodesic: {geo_error:.6f}")

            if metrics["dirichlet"]:
                dirichlet = compute_dirichlet_energy(mesh1, mesh2, p2p)
                results["knn"]["dirichlet"][source_idx, target_idx] = dirichlet
                print(f"KNN - Dirichlet: {dirichlet:.6f}")

            if metrics["coverage"]:
                cov = compute_coverage(p2p, len(mesh2.vertices))
                results["knn"]["coverage"][source_idx, target_idx] = cov
                print(f"KNN - Coverage: {cov:.6f}")

        # 4. Functional Maps
        if methods["fmaps"]:
            p2p = compute_p2p_with_fmaps(
                mesh1_path, mesh2_path, source_features, target_features
            )

            # Evaluate with different metrics
            if metrics["euclidean"]:
                eucl_error = torch.mean(
                    torch.norm(v2.cpu()[p2p][corr_a] - v2.cpu()[corr_b], dim=-1)
                ).item()
                results["fmaps"]["euclidean"][source_idx, target_idx] = eucl_error
                print(f"FUNCTIONAL MAPS - Euclidean: {eucl_error:.6f}")

            if metrics["geodesic"]:
                geo_error = compute_geodesic_error(dist, p2p, corr_a, corr_b)
                results["fmaps"]["geodesic"][source_idx, target_idx] = geo_error
                print(f"FUNCTIONAL MAPS - Geodesic: {geo_error:.6f}")

            if metrics["dirichlet"]:
                dirichlet = compute_dirichlet_energy(mesh1, mesh2, p2p)
                results["fmaps"]["dirichlet"][source_idx, target_idx] = dirichlet
                print(f"FUNCTIONAL MAPS - Dirichlet: {dirichlet:.6f}")

            if metrics["coverage"]:
                cov = compute_coverage(p2p, len(mesh2.vertices))
                results["fmaps"]["coverage"][source_idx, target_idx] = cov
                print(f"FUNCTIONAL MAPS - Coverage: {cov:.6f}")

        # 5. Optimal Transport
        if methods["ot"]:
            p2p = compute_p2p_with_ot(
                source_features.cpu().numpy(), target_features.cpu().numpy()
            )

            # Evaluate with different metrics
            if metrics["euclidean"]:
                eucl_error = torch.mean(
                    torch.norm(v2.cpu()[p2p][corr_a] - v2.cpu()[corr_b], dim=-1)
                ).item()
                results["ot"]["euclidean"][source_idx, target_idx] = eucl_error
                print(f"OPTIMAL TRANSPORT - Euclidean: {eucl_error:.6f}")

            if metrics["geodesic"]:
                geo_error = compute_geodesic_error(dist, p2p, corr_a, corr_b)
                results["ot"]["geodesic"][source_idx, target_idx] = geo_error
                print(f"OPTIMAL TRANSPORT - Geodesic: {geo_error:.6f}")

            if metrics["dirichlet"]:
                dirichlet = compute_dirichlet_energy(mesh1, mesh2, p2p)
                results["ot"]["dirichlet"][source_idx, target_idx] = dirichlet
                print(f"OPTIMAL TRANSPORT - Dirichlet: {dirichlet:.6f}")

            if metrics["coverage"]:
                cov = compute_coverage(p2p, len(mesh2.vertices))
                results["ot"]["coverage"][source_idx, target_idx] = cov
                print(f"OPTIMAL TRANSPORT - Coverage: {cov:.6f}")

        # 6. Flow Composition with ZoomOut
        if methods["flows_composition_zoomout"]:
            p2p = compute_p2p_with_flows_composition_zoomout(
                mesh1_path,
                mesh2_path,
                source_features,
                target_features,
                source_model,
                target_model,
                device,
            )

            # Evaluate with different metrics
            if metrics["euclidean"]:
                eucl_error = torch.mean(
                    torch.norm(v2.cpu()[p2p][corr_a] - v2.cpu()[corr_b], dim=-1)
                ).item()
                results["flows_composition_zoomout"]["euclidean"][
                    source_idx, target_idx
                ] = eucl_error
                print(f"FLOWS COMPOSITION ZOOMOUT - Euclidean: {eucl_error:.6f}")

            if metrics["geodesic"]:
                geo_error = compute_geodesic_error(dist, p2p, corr_a, corr_b)
                results["flows_composition_zoomout"]["geodesic"][
                    source_idx, target_idx
                ] = geo_error
                print(f"FLOWS COMPOSITION ZOOMOUT - Geodesic: {geo_error:.6f}")

            if metrics["dirichlet"]:
                dirichlet = compute_dirichlet_energy(mesh1, mesh2, p2p)
                results["flows_composition_zoomout"]["dirichlet"][
                    source_idx, target_idx
                ] = dirichlet
                print(f"FLOWS COMPOSITION ZOOMOUT - Dirichlet: {dirichlet:.6f}")

            if metrics["coverage"]:
                cov = compute_coverage(p2p, len(mesh2.vertices))
                results["flows_composition_zoomout"]["coverage"][
                    source_idx, target_idx
                ] = cov
                print(f"FLOWS COMPOSITION ZOOMOUT - Coverage: {cov:.6f}")

        # 7. KNN with Neural ZoomOut
        if methods["knn_neural_zoomout"]:
            p2p = compute_p2p_with_knn_neural_zoomout(
                mesh1_path,
                mesh2_path,
                source_features,
                target_features,
            )

            # Evaluate with different metrics
            if metrics["euclidean"]:
                eucl_error = torch.mean(
                    torch.norm(v2.cpu()[p2p][corr_a] - v2.cpu()[corr_b], dim=-1)
                ).item()
                results["knn_neural_zoomout"]["euclidean"][source_idx, target_idx] = (
                    eucl_error
                )
                print(f"KNN NEURAL ZOOMOUT - Euclidean: {eucl_error:.6f}")

            if metrics["geodesic"]:
                geo_error = compute_geodesic_error(dist, p2p, corr_a, corr_b)
                results["knn_neural_zoomout"]["geodesic"][source_idx, target_idx] = (
                    geo_error
                )
                print(f"KNN NEURAL ZOOMOUT - Geodesic: {geo_error:.6f}")

            if metrics["dirichlet"]:
                dirichlet = compute_dirichlet_energy(mesh1, mesh2, p2p)
                results["knn_neural_zoomout"]["dirichlet"][source_idx, target_idx] = (
                    dirichlet
                )
                print(f"KNN NEURAL ZOOMOUT - Dirichlet: {dirichlet:.6f}")

            if metrics["coverage"]:
                cov = compute_coverage(p2p, len(mesh2.vertices))
                results["knn_neural_zoomout"]["coverage"][source_idx, target_idx] = cov
                print(f"KNN NEURAL ZOOMOUT - Coverage: {cov:.6f}")

        # 8. Functional Maps with Neural ZoomOut
        if methods["fmap_neural_zoomout"]:
            p2p = compute_p2p_with_fmap_neural_zoomout(
                mesh1_path,
                mesh2_path,
                source_features,
                target_features,
            )

            # Evaluate with different metrics
            if metrics["euclidean"]:
                eucl_error = torch.mean(
                    torch.norm(v2.cpu()[p2p][corr_a] - v2.cpu()[corr_b], dim=-1)
                ).item()
                results["fmap_neural_zoomout"]["euclidean"][source_idx, target_idx] = (
                    eucl_error
                )
                print(f"FMAP NEURAL ZOOMOUT - Euclidean: {eucl_error:.6f}")

            if metrics["geodesic"]:
                geo_error = compute_geodesic_error(dist, p2p, corr_a, corr_b)
                results["fmap_neural_zoomout"]["geodesic"][source_idx, target_idx] = (
                    geo_error
                )
                print(f"FMAP NEURAL ZOOMOUT - Geodesic: {geo_error:.6f}")

            if metrics["dirichlet"]:
                dirichlet = compute_dirichlet_energy(mesh1, mesh2, p2p)
                results["fmap_neural_zoomout"]["dirichlet"][source_idx, target_idx] = (
                    dirichlet
                )
                print(f"FMAP NEURAL ZOOMOUT - Dirichlet: {dirichlet:.6f}")

            if metrics["coverage"]:
                cov = compute_coverage(p2p, len(mesh2.vertices))
                results["fmap_neural_zoomout"]["coverage"][source_idx, target_idx] = cov
                print(f"FMAP NEURAL ZOOMOUT - Coverage: {cov:.6f}")

        # 9. Neural Deformation Pyramid with Landmarks
        if methods["ndp_landmarks"]:
            # Create TriangleMesh objects for NDP
            if len(mesh1.faces) > 0 and len(mesh2.faces) > 0:
                source_shape = TriangleMesh(
                    np.array(mesh1.vertices), np.array(mesh1.faces)
                )
                target_shape = TriangleMesh(
                    np.array(mesh2.vertices), np.array(mesh2.faces)
                )

                p2p = ndp_with_ldmks(
                    source_shape,
                    target_shape,
                    source_landmarks,
                    target_landmarks,
                )

                # Evaluate with different metrics
                if metrics["euclidean"]:
                    eucl_error = torch.mean(
                        torch.norm(v2.cpu()[p2p][corr_a] - v2.cpu()[corr_b], dim=-1)
                    ).item()
                    results["ndp_landmarks"]["euclidean"][source_idx, target_idx] = (
                        eucl_error
                    )
                    print(f"NDP LANDMARKS - Euclidean: {eucl_error:.6f}")

                if metrics["geodesic"]:
                    geo_error = compute_geodesic_error(dist, p2p, corr_a, corr_b)
                    results["ndp_landmarks"]["geodesic"][source_idx, target_idx] = (
                        geo_error
                    )
                    print(f"NDP LANDMARKS - Geodesic: {geo_error:.6f}")

                if metrics["dirichlet"]:
                    dirichlet = compute_dirichlet_energy(mesh1, mesh2, p2p)
                    results["ndp_landmarks"]["dirichlet"][source_idx, target_idx] = (
                        dirichlet
                    )
                    print(f"NDP LANDMARKS - Dirichlet: {dirichlet:.6f}")

                if metrics["coverage"]:
                    cov = compute_coverage(p2p, len(mesh2.vertices))
                    results["ndp_landmarks"]["coverage"][source_idx, target_idx] = cov
                    print(f"NDP LANDMARKS - Coverage: {cov:.6f}")
            else:
                print(
                    "Warning: NDP method requires mesh faces, skipping for point clouds."
                )

        # 10. Functional Maps with ZoomOut
        if methods["fmap_zoomout"]:
            p2p = compute_p2p_with_fmap_zoomout(
                mesh1_path, mesh2_path, source_features, target_features
            )

            # Evaluate with different metrics
            if metrics["euclidean"]:
                eucl_error = torch.mean(
                    torch.norm(v2.cpu()[p2p][corr_a] - v2.cpu()[corr_b], dim=-1)
                ).item()
                results["fmap_zoomout"]["euclidean"][source_idx, target_idx] = (
                    eucl_error
                )
                print(f"FMAP ZOOMOUT - Euclidean: {eucl_error:.6f}")

            if metrics["geodesic"]:
                geo_error = compute_geodesic_error(dist, p2p, corr_a, corr_b)
                results["fmap_zoomout"]["geodesic"][source_idx, target_idx] = geo_error
                print(f"FMAP ZOOMOUT - Geodesic: {geo_error:.6f}")

            if metrics["dirichlet"]:
                dirichlet = compute_dirichlet_energy(mesh1, mesh2, p2p)
                results["fmap_zoomout"]["dirichlet"][source_idx, target_idx] = dirichlet
                print(f"FMAP ZOOMOUT - Dirichlet: {dirichlet:.6f}")

            if metrics["coverage"]:
                cov = compute_coverage(p2p, len(mesh2.vertices))
                results["fmap_zoomout"]["coverage"][source_idx, target_idx] = cov
                print(f"FMAP ZOOMOUT - Coverage: {cov:.6f}")

        # 11. KNN with ZoomOut
        if methods["knn_zoomout"]:
            p2p = compute_p2p_with_knn_zoomout(
                mesh1_path, mesh2_path, source_features, target_features
            )

            # Evaluate with different metrics
            if metrics["euclidean"]:
                eucl_error = torch.mean(
                    torch.norm(v2.cpu()[p2p][corr_a] - v2.cpu()[corr_b], dim=-1)
                ).item()
                results["knn_zoomout"]["euclidean"][source_idx, target_idx] = eucl_error
                print(f"KNN ZOOMOUT - Euclidean: {eucl_error:.6f}")

            if metrics["geodesic"]:
                geo_error = compute_geodesic_error(dist, p2p, corr_a, corr_b)
                results["knn_zoomout"]["geodesic"][source_idx, target_idx] = geo_error
                print(f"KNN ZOOMOUT - Geodesic: {geo_error:.6f}")

            if metrics["dirichlet"]:
                dirichlet = compute_dirichlet_energy(mesh1, mesh2, p2p)
                results["knn_zoomout"]["dirichlet"][source_idx, target_idx] = dirichlet
                print(f"KNN ZOOMOUT - Dirichlet: {dirichlet:.6f}")

            if metrics["coverage"]:
                cov = compute_coverage(p2p, len(mesh2.vertices))
                results["knn_zoomout"]["coverage"][source_idx, target_idx] = cov
                print(f"KNN ZOOMOUT - Coverage: {cov:.6f}")

        # 12. Functional Maps with WKS
        if methods["fmaps_wks"]:
            # Use correspondence points as landmarks for WKS

            p2p = compute_p2p_with_fmaps_wks(
                mesh1_path, mesh2_path, source_landmarks, target_landmarks
            )

            # Evaluate with different metrics
            if metrics["euclidean"]:
                eucl_error = torch.mean(
                    torch.norm(v2.cpu()[p2p][corr_a] - v2.cpu()[corr_b], dim=-1)
                ).item()
                results["fmaps_wks"]["euclidean"][source_idx, target_idx] = eucl_error
                print(f"FMAPS WKS - Euclidean: {eucl_error:.6f}")

            if metrics["geodesic"]:
                geo_error = compute_geodesic_error(dist, p2p, corr_a, corr_b)
                results["fmaps_wks"]["geodesic"][source_idx, target_idx] = geo_error
                print(f"FMAPS WKS - Geodesic: {geo_error:.6f}")

            if metrics["dirichlet"]:
                dirichlet = compute_dirichlet_energy(mesh1, mesh2, p2p)
                results["fmaps_wks"]["dirichlet"][source_idx, target_idx] = dirichlet
                print(f"FMAPS WKS - Dirichlet: {dirichlet:.6f}")

            if metrics["coverage"]:
                cov = compute_coverage(p2p, len(mesh2.vertices))
                results["fmaps_wks"]["coverage"][source_idx, target_idx] = cov
                print(f"FMAPS WKS - Coverage: {cov:.6f}")

    # Print mean results
    print("\n--- FINAL RESULTS ---")
    summary_results = {}

    # Compute mean values for each method and metric
    for method_name in [m for m, enabled in methods.items() if enabled]:
        summary_results[method_name] = {}

        for metric_name in [m for m, enabled in metrics.items() if enabled]:
            result_matrix = results[method_name][metric_name]
            mean_value = (
                np.mean(result_matrix[result_matrix > 0])
                if np.any(result_matrix > 0)
                else 0
            )
            summary_results[method_name][metric_name] = float(mean_value)
            print(f"{method_name.upper()} - {metric_name.upper()}: {mean_value:.6f}")

    # Save individual result matrices
    for method_name in [m for m, enabled in methods.items() if enabled]:
        for metric_name in [m for m, enabled in metrics.items() if enabled]:
            np.save(
                os.path.join(out_dir, f"results_{method_name}_{metric_name}.npy"),
                results[method_name][metric_name],
            )

    # Save summary to JSON
    summary = {
        "config_path": args.config,
        "num_shapes": len(file_names),
        "shape_names": file_names,
        "methods": [m for m, enabled in methods.items() if enabled],
        "metrics": [m for m, enabled in metrics.items() if enabled],
        "results": summary_results,
    }

    with open(os.path.join(out_dir, "results_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    args = parse_arguments()

    # Load configuration file
    with open(args.config, "r") as f:
        config = json.load(f)

    run_matching_experiments(config, args)
