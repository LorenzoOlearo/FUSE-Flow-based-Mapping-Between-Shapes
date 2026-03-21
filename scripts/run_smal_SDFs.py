import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import List

import numpy as np


def get_targets(output_dir, dataset_dir, overwrite) -> List[str]:
    targets = []

    for file in os.listdir(dataset_dir):
        if file.endswith(".off"):
            shape_name = os.path.splitext(file)[0]
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(Path(output_dir, shape_name), exist_ok=True)

            if (
                not os.path.exists(Path(output_dir, shape_name, "checkpoint-9999.pth"))
                or overwrite
            ):
                targets.append(shape_name)

    return targets


def main(args):
    with open(args.config) as f:
        config = json.load(f)
    matching_config = config["matching_config"]["SMAL"]

    dataset_dir = Path(matching_config["dataset_path"])
    corr_path = matching_config["corr_path"]
    base_landmarks = np.array(matching_config["landmarks"])
    output_dir = Path(matching_config["flows_SDFs_path"])

    if args.run_name is not None:
        output_dir = Path(matching_config["flows_SDFs_path"]).parent / args.run_name

    targets = get_targets(output_dir, dataset_dir, overwrite=args.overwrite)
    if targets == []:
        print("No targets found, exiting.")
        exit(0)
    print(f"Processing {len(targets)} targets: {targets}")

    for target in targets:
        target_dir = Path(output_dir, target)

        working_dir = Path(str(Path(__file__).resolve()).split("/scripts")[0])
        data_path = Path(working_dir, dataset_dir, f"{target}.off")
        features_path = Path(
            working_dir, "data", "SMAL_features_pca_20", f"{target}_features.npy"
        )

        corr = np.array(np.loadtxt(f"{corr_path}/{target}.vts")) - 1
        target_landmarks = corr[base_landmarks].tolist()

        config = {
            "device": "cuda:1",
            "blr": 5e-7,
            "output_dir": str(target_dir),
            "log_dir": str(target_dir),
            "data_path": str(data_path),
            "train": True,
            "inference": True,
            "epochs": 10000,
            "num_steps": 64,
            "method": "FM",
            "network": "MLP",
            "edm_preconditioning": False,
            "mlp_hidden_size": 256,
            "mlp_depth": 4,
            "mlp_num_frequencies": -1,
            "batch_size": 50000,
            "num_points_train": 50000,
            "learning_rate": 0.01,
            "distribution": "gaussian",
            "embedding_dim": len(target_landmarks),
            "embedding_type": "features_only",
            "features_type": "landmarks",
            "landmarks": target_landmarks,
        }

        config_path = os.path.join(target_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)

        if args.external is True:
            command = [
                "python",
                "main.py",
                "--config",
                config_path,
                "--features_path",
                str(features_path),
                "--features_interpolation",
                str(500000),
            ]
        else:
            command = [
                "python",
                "main.py",
                "--config",
                config_path,
                "--features_normalization",
                "0_center_indipendent",
            ]

        command_str = " ".join(command)
        print(f"Running command: {command_str}")

        try:
            subprocess.run(command, check=True)
            print(f"Successfully processed {target}")
        except subprocess.CalledProcessError as e:
            print(f"Error processing {target}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a flow on all SMAL meshes")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the matching config JSON (provides all dataset and output paths)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help='Overwrite if an existing flow model "checkpoint-9999.pth" is found',
        default="False",
    )
    parser.add_argument(
        "--external",
        action="store_true",
        help="Use external precomputed features",
        default="False",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Override the output directory; saves to matching_config.SMAL.flows_SDFs_path/../<run_name>/",
    )

    args = parser.parse_args()

    print("Training flows on SMAL dataset:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print("-----------------------------------------------")

    main(args)
