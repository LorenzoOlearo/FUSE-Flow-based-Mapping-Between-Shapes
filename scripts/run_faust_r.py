import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import List

import numpy as np

OUTPUT_DIR = Path("./out/flows/faust_r/faust_r/")
FAUST_R_DIR = Path("./data/FAUST_r/off")


def get_targets(overwrite) -> List[str]:
    targets = []

    for i in range(80, 100):
        target = f"tr_reg_{i:03d}"
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.makedirs(Path(OUTPUT_DIR, target), exist_ok=True)

        if (
            os.path.exists(Path(OUTPUT_DIR, target, "checkpoint-best.pth"))
            and overwrite == True
        ):
            targets.append(target)
        elif not os.path.exists(Path(OUTPUT_DIR, target, "checkpoint-best.pth")):
            targets.append(target)

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
        data_path = Path(working_dir, FAUST_R_DIR, f"{target}.off")
        features_path = Path(
            working_dir, "data", "FAUST_R_features_pca_20", f"{target}_features.npy"
        )

        faust_r_landmarks = ([2650, 3663, 3089, 1979, 1078],)
        corr = np.array(np.loadtxt(f"./data/FAUST_r/corres/{target}.vts")) - 1
        target_landmarks = corr[faust_r_landmarks].astype(int)

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
            "distribution": "gaussian",
            "embedding_dim": 20,
            "embedding_type": "features_only",
            "features_type": "wks_landmarks",
            "use_heat_method": False,
            "features_normalization": "none",
            "dists_path": "./data/FAUST_R/training/registrations/dists/",
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
