import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import List

OUTPUT_DIR = Path("./out/flows/tosca-sym/tosca-sym-baseline/")
TOSCA_DIR = Path("./data/TOSCA/TOSCAhires/")


def get_targets(overwrite, target) -> List[str]:
    targets = []

    if target is not None:
        target_files = [
            f
            for f in os.listdir(TOSCA_DIR)
            if f.startswith(target) and f.endswith(".ply")
        ]
        print(f"Found {len(target_files)} files for target {target}: {target_files}")
        for file in target_files:
            target = file.split(".ply")[0]
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            os.makedirs(Path(OUTPUT_DIR, target), exist_ok=True)

            if (
                os.path.exists(Path(OUTPUT_DIR, target, "checkpoint-9999.pth"))
                and overwrite == True
            ):
                targets.append(target)
            elif not os.path.exists(Path(OUTPUT_DIR, target, "checkpoint-9999.pth")):
                targets.append(target)
    else:
        for file in os.listdir(TOSCA_DIR):
            if file.endswith(".ply"):
                target = file.split(".ply")[0]
                os.makedirs(OUTPUT_DIR, exist_ok=True)
                os.makedirs(Path(OUTPUT_DIR, target), exist_ok=True)

                if (
                    os.path.exists(Path(OUTPUT_DIR, target, "checkpoint-9999.pth"))
                    and overwrite == True
                ):
                    targets.append(target)
                elif not os.path.exists(
                    Path(OUTPUT_DIR, target, "checkpoint-9999.pth")
                ):
                    targets.append(target)

    return targets


def main(args):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    targets = get_targets(args.overwrite, args.target)
    if targets == []:
        print("No targets found, exiting.")
        exit(0)

    print(f"Processing {len(targets)} targets: {targets}")

    for target in targets:
        target_dir = Path(OUTPUT_DIR, target)
        os.makedirs(target_dir, exist_ok=True)

        working_dir = Path(str(Path(__file__).resolve()).split("/scripts")[0])
        data_path = Path(working_dir, TOSCA_DIR, f"{target}.ply")
        features_path = Path(
            working_dir, "data", "TOSCA_features_pca_20", f"{target}_features.npy"
        )

        if "cat" in target:
            # [noose tip, left front foot bottom, right front foot bottom, left back foot bottom, right back foot bottom, tail tip slightly above the tail tip]
            landmarks = [8633, 16393, 19336, 22547, 26685, 27798]
        else:
            print(f"[OCIO] Landmarks not defined for target {target}")
            landmarks = [-1, -1, -1, -1, -1, -1]

        config = {
            "device": "cuda:0",
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
            "learning_rate": 0.01,
            "distribution": "gaussian",
            "embedding_dim": 20,
            "embedding_type": "features_only",
            "features_type": "wks",
            "use_heat_method": False,
            "distribution": "gaussian",
            "features_normalization": "0_center_global",
            "dists_path": "./data/TOSCA/TOSCAhires/dists/",
            "landmarks": landmarks,
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
                str(100_000),
            ]

        command_str = " ".join(command)
        print(f"Running command: {command_str}")

        try:
            subprocess.run(command, check=True)
            print(f"Successfully processed {target}")
        except subprocess.CalledProcessError as e:
            print(f"Error processing {target}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a flow on TOSCA meshes")
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
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Target to process (cat, centaur, david, dog, gorilla, horse, michael, victoria, wolf), if None, process all targets",
    )

    args = parser.parse_args()

    print("Training flows on FAUST dataset:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print("-----------------------------------------------")

    main(args)
