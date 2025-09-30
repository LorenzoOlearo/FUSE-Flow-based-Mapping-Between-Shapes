import os
import json
import subprocess
import argparse
import re

from pathlib import Path
from typing import List

OUTPUT_DIR = Path('./out/flows/faust-SDFs/faust-SDFs-diameter-norm')
SDF_DIR = Path('./out/SDFs')
FAUST_DIR = Path('./data/MPI-FAUST/training/registrations')


def get_targets(overwrite, test) -> List[str]:
    """Process all SDFs on which mesh vertex distances have been computed"""
    if overwrite and test:
        targets = [
            f.name for f in SDF_DIR.iterdir() if f.is_dir() and
            any(80 <= int(num) <= 99 for num in re.findall(r'\d+', f.name)) and
            any(child.suffix == '.pth' for child in f.iterdir()) and
            any(child.name.endswith('sdf-dijkstra-surface-points.txt') for child in f.iterdir())
        ]
    elif overwrite == True:
        targets = [
            f.name for f in SDF_DIR.iterdir() if f.is_dir() and
            any(child.suffix == '.pth' for child in f.iterdir()) and
            any(child.name.endswith('sdf-dijkstra-surface-points.txt') for child in f.iterdir())
        ]
    elif test == True:
        targets = [
            f.name for f in SDF_DIR.iterdir() if f.is_dir() and
            any(80 <= int(num) <= 99 for num in re.findall(r'\d+', f.name)) and
            any(child.suffix == '.pth' for child in f.iterdir()) and
            any(child.name.endswith('sdf-dijkstra-surface-points.txt') for child in f.iterdir())
        ]
    else:
        targets = []
        for f in SDF_DIR.iterdir():
            if not f.is_dir():
                continue
            has_pth = any(child.suffix == '.pth' for child in f.iterdir())
            has_sdf = any(child.name.endswith('sdf-dijkstra-surface-points.txt') for child in f.iterdir())
            if not (has_pth and has_sdf):
                continue
            checkpoint = Path(OUTPUT_DIR, f.name, 'checkpoint-9999.pth')
            if checkpoint.exists():
                continue
            targets.append(f.name)
    return targets


def main(args):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    targets = get_targets(args.overwrite, args.test)
    if targets == []:
        print("No targets found, exiting.")
        exit(0)

    print(f"Processing {len(targets)} targets: {targets}")

    for target in targets:
        target_dir = Path(OUTPUT_DIR, target)
        os.makedirs(target_dir, exist_ok=True)

        # Get working directory
        working_dir = Path(str(Path(__file__).resolve()).split('/scripts')[0])
        data_path = Path(working_dir, FAUST_DIR, f"{target}.ply")

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
            "batch_size": 100000,
            "num_points_train": 500000,
            "learning_rate": 0.01,
            "distribution": "gaussian",
            "embedding_dim": 5,
            "embedding_type": "features_only",
            "features_type": "landmarks",
            "features_normalization": "diameter",
            "landmarks": [412, 5891, 6593, 3323, 2119]
        }

        config_path = os.path.join(target_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)

        command = [
            "python", "main.py",
            "--config", config_path,
            "--features_path", f"{SDF_DIR}/{target}/{target}-sdf-dijkstra-features.txt",
        ]

        command_str = " ".join(command)
        print(f"Running command: {command_str}")

        try:
            subprocess.run(command, check=True)
            print(f"Successfully processed {target}")
        except subprocess.CalledProcessError as e:
            print(f"Error processing {target}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a flow on all FAUST SDFs features")
    parser.add_argument('--overwrite', action='store_true', help="Overwrite if an existing flow model \"checkpoint-9999.pth\" is found", default='False')
    parser.add_argument('--test', action='store_true', help="Test mode, only process targets that are not in the range 80-99", default='False')

    args = parser.parse_args()

    print("Training flows on FAUST SDFs features:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print("-----------------------------------------------")

    main(args)
