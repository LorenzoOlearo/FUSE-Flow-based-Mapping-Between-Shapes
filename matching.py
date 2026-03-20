import os

os.environ["GEOMSTATS_BACKEND"] = "pytorch"

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from matching.data_structures import DataPath
from matching.pipeline import process_pair
from matching.targets import (
    get_targets_faust,
    get_targets_faust_r,
    get_targets_kinect,
    get_targets_shrec19,
    get_targets_shrec20,
    get_targets_smal,
    get_targets_smplx,
    get_targets_surreal,
    get_targets_tosca,
)


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
            raise ValueError(
                "The 'pt' representation is only supported for the KINECT dataset."
            )
        dataset = "SMAL"
        targets = get_targets_smal(
            Path(config["matching_config"][dataset]["flows_path"])
        )
    elif args.surreal:
        if args.source_rep == "pt" or args.target_rep == "pt":
            raise ValueError(
                "The 'pt' representation is only supported for the KINECT dataset."
            )
        dataset = "SURREAL"
        targets = get_targets_surreal(
            Path(config["matching_config"][dataset]["flows_path"])
        )
    elif args.kinect:
        dataset = "KINECT"
        targets = get_targets_kinect(
            Path(config["matching_config"][dataset]["flows_path"])
        )
    elif args.smplx:
        dataset = "SMPLX"
        targets = get_targets_smplx(
            Path(config["matching_config"][dataset]["flows_path"])
        )
    elif args.shrec20:
        dataset = "SHREC20"
        targets = get_targets_shrec20(
            Path(config["matching_config"][dataset]["flows_path"])
        )
    elif args.shrec19:
        dataset = "SHREC19"
        targets = get_targets_shrec19(
            Path(config["matching_config"][dataset]["flows_path"])
        )
    elif args.tosca:
        dataset = "TOSCA"
        targets = get_targets_tosca(
            Path(config["matching_config"][dataset]["flows_path"])
        )
    elif args.faust_r:
        dataset = "FAUST_R"
        targets = get_targets_faust_r(args)
    else:
        raise ValueError(
            "Please specify either --faust, --smal, --kinect, --surreal, --smplx, --shrec20, or --shrec19, tosca to select the dataset."
        )

    if targets is None or len(targets) == 0:
        raise ValueError("No targets found to process.")
    tqdm.write(f"Running {dataset} experiments")
    tqdm.write(f"Found {len(targets)} targets: {targets}")

    output_dir = Path(
        f"./out/matching/{dataset.lower()}/matching-{args.source_rep}-{args.target_rep}-{args.matching_run_name}"
    )
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
        flows_SDFs_path=(
            Path(config["matching_config"][dataset]["flows_SDFs_path"])
            if "flows_SDFs_path" in config["matching_config"][dataset]
            else None
        ),
        sdf_path=(
            Path(config["matching_config"][dataset]["SDFs_path"])
            if "SDFs_path" in config["matching_config"][dataset]
            else None
        ),
        corr_path=(
            Path(config["matching_config"][dataset]["corr_path"])
            if "corr_path" in config["matching_config"][dataset]
            else None
        ),
        corr_offset=(
            config["matching_config"][dataset]["corr_offset"]
            if "corr_offset" in config["matching_config"][dataset]
            else 0
        ),
        common_landmarks_path=(
            Path(config["matching_config"][dataset]["common_landmarks_path"])
            if dataset == "SHREC20"
            else None
        ),
        gts_path=(
            Path(config["matching_config"][dataset]["gts_path"])
            if "gts_path" in config["matching_config"][dataset]
            else None
        ),
        scan_flows_path=(
            Path(config["matching_config"][dataset]["scan_flows_path"])
            if "scan_flows_path" in config["matching_config"][dataset]
            else None
        ),
        scan_dataset_path=(
            Path(config["matching_config"][dataset]["scan_dataset_path"])
            if "scan_dataset_path" in config["matching_config"][dataset]
            else None
        ),
        scan_dists_path=(
            Path(config["matching_config"][dataset]["scan_dists_path"])
            if "scan_dists_path" in config["matching_config"][dataset]
            else None
        ),
        scan_features_path=(
            Path(config["matching_config"][dataset]["scan_features_path"])
            if "scan_features_path" in config["matching_config"][dataset]
            else None
        ),
        scan_dataset_extension=(
            config["matching_config"][dataset]["scan_dataset_extension"]
            if "scan_dataset_extension" in config["matching_config"][dataset]
            else None
        ),
        scan_corr_path=(
            Path(config["matching_config"][dataset]["scan_corr_path"])
            if "scan_corr_path" in config["matching_config"][dataset]
            else None
        ),
    )

    results = []
    times = []
    results_file = Path(output_dir, "matching_results.csv")

    if args.same:
        pairs = [(t, t) for t in targets]

    elif args.pt_skinning and dataset == "KINECT":
        sources = get_targets_smplx(
            Path(config["matching_config"]["SMPLX"]["smplx_template_flows_path"])
        )
        targets = get_targets_kinect(
            Path(config["matching_config"][dataset]["flows_path"])
        )
        pairs = [(s, t) for s in sources for t in targets]

    elif dataset == "SHREC19":
        # Only load the pairs defined in the corr_path
        if data_path.corr_path is None:
            raise ValueError(
                "For SHREC19 dataset, corr_path must be provided in the config to define the matching pairs."
            )
        corr_files = [f for f in os.listdir(data_path.corr_path) if f.endswith(".txt")]
        pairs = []
        for f in corr_files:
            base = os.path.splitext(f)[0]
            parts = base.split("_")
            if len(parts) != 2:
                tqdm.write(
                    f"Skipping file with unexpected format (expected 'target_source.txt'): {f}"
                )
                continue
            source, target = parts[0], parts[1]
            if source in targets and target in targets:
                if source != "43" and target != "43":
                    pairs.append((source, target))
    else:
        # Default case in which we match all different pairs
        pairs = [(s, t) for s in targets for t in targets if s != t]

    # Override pairs if specific source/target shapes are provided
    if args.source_shape is not None and args.target_shape is not None:
        if args.source_shape not in targets:
            raise ValueError(
                f"Source shape '{args.source_shape}' not found in the targets list."
            )
        if args.target_shape not in targets:
            raise ValueError(
                f"Target shape '{args.target_shape}' not found in the targets list."
            )
        pairs = [(args.source_shape, args.target_shape)]
    elif args.source_shape is not None:
        if args.source_shape not in targets:
            raise ValueError(
                f"Source shape '{args.source_shape}' not found in the targets list."
            )
        pairs = [(args.source_shape, t) for t in targets if t != args.source_shape]
    elif args.target_shape is not None:
        if args.target_shape not in targets:
            raise ValueError(
                f"Target shape '{args.target_shape}' not found in the targets list."
            )
        pairs = [(s, args.target_shape) for s in targets if s != args.target_shape]

    tqdm.write(f"Processing {len(pairs)} shape pairs.")
    for source, target in tqdm(
        pairs, desc="Processing shape pairs", unit="pair", dynamic_ncols=True
    ):
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
            backward_steps=args.backward_steps,
            forward_steps=args.forward_steps,
            embedding_dim=config["embedding_dim"],
            mlp_hidden_size=args.mlp_hidden_size,
            mlp_depth=args.mlp_depth,
            mlp_num_frequencies=args.mlp_num_frequencies,
        )
        elapsed_time = time.perf_counter() - start_time
        tqdm.write(f"Time taken for {source} -> {target}: {elapsed_time:.2f} seconds")
        times.append(elapsed_time)

        df.to_csv(results_file, mode="a", header=not results_file.exists(), index=False)
        results.append(df)

    results_df = pd.concat(results, ignore_index=True)
    results_df.to_csv(Path(output_dir, "matching_results_completed.csv"), index=False)

    times_sec = [
        t.total_seconds() if hasattr(t, "total_seconds") else float(t) for t in times
    ]
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
        "--tosca",
        action="store_true",
        help="Run matching methods on the TOSCA SYM dataset",
        default=False,
    )
    parser.add_argument(
        "--faust_r",
        action="store_true",
        help="Run matching methods on the TOSCA_R SYM dataset",
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
        default=0,
        type=int,
        help="offset to apply to the correspondences, in case of SMAL offset : -1, otherwise default=0",
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
    parser.add_argument(
        "--backward_steps",
        type=int,
        default=64,
        help="Number of backward steps for flow-based matching methods",
    )
    parser.add_argument(
        "--forward_steps",
        type=int,
        default=64,
        help="Number of forward steps for flow-based matching methods",
    )
    parser.add_argument(
        "--mlp_hidden_size", default=256, type=int, help="Hidden size of the MLP"
    )
    parser.add_argument(
        "--mlp_depth", default=4, type=int, help="Depth (number of layers) of the MLP"
    )
    parser.add_argument(
        "--mlp_num_frequencies",
        default=-1,
        type=int,
        help="Number of Fourier frequencies for the MLP input encoding (-1 to disable)",
    )

    # First pass: extract --config path only
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=None, type=str)
    pre_args, _ = pre_parser.parse_known_args()

    # Config sets new defaults; CLI args will still override them
    if pre_args.config:
        with open(pre_args.config, "r") as f:
            config = json.load(f)
        known_dests = {action.dest for action in parser._actions}
        for key in config:
            if key not in known_dests:
                tqdm.write(
                    f"WARNING: config key '{key}' does not match any argument and will be ignored"
                )
        valid_config = {k: v for k, v in config.items() if k in known_dests}
        parser.set_defaults(**valid_config)

    args = parser.parse_args()

    if args.matching_run_name == "":
        args.matching_run_name = input("Enter a run name: ")

    tqdm.write("------------------------------------------")
    for arg in vars(args):
        tqdm.write(f"{arg}: {getattr(args, arg)}")
    tqdm.write("------------------------------------------")

    main(args)
