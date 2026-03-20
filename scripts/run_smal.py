import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import List

import numpy as np

OUTPUT_DIR = Path("./out/flows/smal/smal-wks-10")
SMAL_DIR = Path("./data/SMAL_r/off")


def get_targets(overwrite) -> List[str]:
    targets = []

    for file in os.listdir(SMAL_DIR):
        if file.endswith(".off") and file.startswith(("cougar", "hippo", "horse")):
            shape_name = os.path.splitext(file)[0]
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            os.makedirs(Path(OUTPUT_DIR, shape_name), exist_ok=True)

            if (
                not os.path.exists(Path(OUTPUT_DIR, shape_name, "checkpoint-best.pth"))
                or overwrite
            ):
                targets.append(shape_name)

    return targets


def main(args):
    targets = get_targets(overwrite=args.overwrite)
    if targets == []:
        print("No targets found, exiting.")
        exit(0)
    print(f"Processing {len(targets)} targets: {targets}")

    for target in targets:
        target_dir = Path(OUTPUT_DIR, target)

        working_dir = Path(str(Path(__file__).resolve()).split("/scripts")[0])
        data_path = Path(working_dir, SMAL_DIR, f"{target}.off")
        vertex_features_path = Path(
            working_dir, "data", "SMAL_r", "smal_precomputed_features", f"{target}.npy"
        )

        smal_landmarks = np.array([3162, 1931, 3731, 1399, 1111, 1001])
        corr = np.array(np.loadtxt(f"./data/SMAL_r/corres/{target}.vts")) - 1
        target_landmarks = corr[smal_landmarks].astype(int)

        config = {
            "device": "cuda:0",
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
            "batch_size": 50_000,
            "num_points_train": 50_000,
            "learning_rate": 0.001,
            "embedding_dim": 10,
            "embedding_type": "features_only",
            "features_type": "wks_landmarks",
            "use_heat_method": False,
            "distribution": "gaussian",
            "features_normalization": "none",
            "dists_path": "./data/SMAL_r/dists/",
            "landmarks": target_landmarks.tolist(),
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
                "--vertex_features_path",
                str(vertex_features_path),
                "--features_interpolation",
                str(100_000),
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
    parser = argparse.ArgumentParser(description="Train a flow on all SMAL meshes")
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
        "--method",
        type=str,
        default="FM",
        help="Method to use to construct the flows: FM or Diffusion (DDIM)",
    )

    args = parser.parse_args()

    print("Training flows on FAUST dataset:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print("-----------------------------------------------")

    main(args)
