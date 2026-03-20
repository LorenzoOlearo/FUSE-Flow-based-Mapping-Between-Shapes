import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import List

# OUTPUT_DIR = Path('./out/flows/faust/faust-diameter-norm-WKS-EXPERIMENTS-NO-FF-BIGGER-but-still-256/')
OUTPUT_DIR = Path("./out/flows/faust/faust-geodesics-17-CHECK/")
FAUST_DIR = Path("./data/MPI-FAUST/training/registrations")


def get_targets(overwrite) -> List[str]:
    targets = []

    for i in range(80, 100):
        target = f"tr_reg_{i:03d}"
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.makedirs(Path(OUTPUT_DIR, target), exist_ok=True)

        if (
            os.path.exists(Path(OUTPUT_DIR, target, "checkpoint-9999.pth"))
            and overwrite == True
        ):
            targets.append(target)
        elif not os.path.exists(Path(OUTPUT_DIR, target, "checkpoint-9999.pth")):
            targets.append(target)

    return targets


def main(args):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    targets = get_targets(args.overwrite)
    if targets == []:
        print("No targets found, exiting.")
        exit(0)

    print(f"Processing {len(targets)} targets: {targets}")

    for target in targets:
        target_dir = Path(OUTPUT_DIR, target)
        os.makedirs(target_dir, exist_ok=True)

        working_dir = Path(str(Path(__file__).resolve()).split("/scripts")[0])
        data_path = Path(working_dir, FAUST_DIR, f"{target}.ply")
        vertex_features_path = Path(
            working_dir,
            "data",
            "MPI-FAUST",
            "training",
            "registrations",
            "faust_precomputed_features",
            f"{target}.npy",
        )

        config = {
            "device": "cuda:1",
            "blr": 5e-7,
            "output_dir": str(target_dir),
            "log_dir": str(target_dir),
            "data_path": str(data_path),
            "train": True,
            "inference": True,
            "epochs": 10_000,
            "num_steps": 64,
            "method": args.method,
            "network": "MLP",
            "batch_size": 50_000,
            "num_points_train": 50_000,
            "learning_rate": 0.001,
            "distribution": "gaussian",
            "embedding_dim": 17,
            "embedding_type": "features_only",
            "features_type": "landmarks",
            "use_heat_method": False,
            "features_normalization": "diameter",
            "dists_path": "./data/MPI-FAUST/training/registrations/dists/",
            "___landmarks": [
                412,
                5891,
                6593,
                3323,
                2119,
                3146,
                1897,
                5260,
                712,
                6416,
                1460,
                4931,
                445,
                6782,
                4933,
                3857,
                3479,
            ],
            "landmarks": [
                412,
                5891,
                6593,
                3323,
                2119,
                3146,
                1897,
                5260,
                712,
                6416,
                1460,
                4931,
                445,
                6782,
                4933,
                3857,
                3479,
            ],
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
    parser = argparse.ArgumentParser(description="Train a flow on all FAUST meshes")
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
