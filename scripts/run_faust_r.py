import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import List

import numpy as np


def get_targets(output_dir, overwrite) -> List[str]:
    targets = []

    for i in range(80, 100):
        target = f"tr_reg_{i:03d}"
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(Path(output_dir, target), exist_ok=True)

        if (
            os.path.exists(Path(output_dir, target, "checkpoint-best.pth"))
            and overwrite == True
        ):
            targets.append(target)
        elif not os.path.exists(Path(output_dir, target, "checkpoint-best.pth")):
            targets.append(target)

    return targets


def main(args):
    with open(args.config) as f:
        config = json.load(f)
    matching_config = config["matching_config"]["FAUST_R"]

    dataset_dir = Path(matching_config["dataset_path"])
    dists_path = matching_config["dists_path"]
    corr_path = matching_config["corr_path"]
    base_landmarks = matching_config["landmarks"]
    output_dir = Path(matching_config["flows_path"])

    if args.run_name is not None:
        output_dir = Path(matching_config["flows_path"]).parent / args.run_name

    targets = get_targets(output_dir, overwrite=args.overwrite)
    if targets == []:
        print("No targets found, exiting.")
        exit(0)
    print(f"Processing {len(targets)} targets: {targets}")

    for target in targets:
        target_dir = Path(output_dir, target)

        working_dir = Path(str(Path(__file__).resolve()).split("/scripts")[0])
        data_path = Path(working_dir, dataset_dir, f"{target}.off")
        features_path = Path(
            working_dir, "data", "FAUST_R_features_pca_20", f"{target}_features.npy"
        )

        corr = np.array(np.loadtxt(f"{corr_path}/{target}.vts")) - 1
        target_landmarks = corr[base_landmarks].astype(int).tolist()

        config = {
            "device": "cuda:1",
            "blr": 5e-7,
            "output_dir": str(target_dir),
            "log_dir": str(target_dir),
            "data_path": str(data_path),
            "train": True,
            "inference": True,
            "epochs": 30_000,
            "num_steps": 64,
            "method": args.method,
            "network": "MLP",
            "edm_preconditioning": False,
            "mlp_hidden_size": 256,
            "mlp_depth": 4,
            "mlp_num_frequencies": -1,
            "batch_size": 50_000,
            "num_points_train": 50_000,
            "learning_rate": 0.0001,
            "lr_scheduler": "cosine",
            "lr_patience": 200,
            "lr_factor": 0.5,
            "distribution": "gaussian",
            "embedding_dim": 20,
            "embedding_type": "features_only",
            "features_type": "wks_landmarks",
            "use_heat_method": False,
            "features_normalization": "none",
            "dists_path": dists_path,
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
                "--features_interpolation",
                str(1_000_000),
            ]

        command_str = " ".join(command)
        print(f"Running command: {command_str}")

        try:
            subprocess.run(command, check=True)
            print(f"Successfully processed {target}")
        except subprocess.CalledProcessError as e:
            print(f"Error processing {target}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a flow on all FAUST_R meshes")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the matching config JSON (provides all dataset and output paths)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help='Overwrite if an existing flow model "checkpoint-best.pth" is found',
        default="False",
    )
    parser.add_argument(
        "--external",
        action="store_true",
        help="Use external precomputed features",
        default="False",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="FM",
        help="Method to use to construct the flows: FM or Diffusion (DDIM)",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Override the output directory; saves to matching_config.FAUST_R.flows_path/../<run_name>/",
    )
    args = parser.parse_args()

    print("Training flows on FAUST_R dataset:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print("-----------------------------------------------")

    main(args)
