import subprocess
import os
import numpy as np
import json
import argparse

from pathlib import Path
from typing import List

OUTPUT_DIR = Path('./out/flows/scan-faust/full-scan-faust-diameter-norm-points/')
SCAN_FAUST_DIR = Path('./data/SCAN-FAUST/full/shapes/')


def get_targets(overwrite) -> List[str]:
    targets = []

    for file in os.listdir(SCAN_FAUST_DIR):
        if file.endswith(".ply"):
            shape_name = os.path.splitext(file)[0]
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            os.makedirs(Path(OUTPUT_DIR, shape_name), exist_ok=True)

            if not os.path.exists(Path(OUTPUT_DIR, shape_name, 'checkpoint-9999.pth')) or overwrite:
                targets.append(shape_name)

    return targets


def get_targets_prioritized(overwrite) -> List[str]:
    targets = []

    for file in os.listdir(SCAN_FAUST_DIR):
        if file.endswith(".ply"):
            shape_name = os.path.splitext(file)[0]
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            os.makedirs(Path(OUTPUT_DIR, shape_name), exist_ok=True)

            if not os.path.exists(Path(OUTPUT_DIR, shape_name, 'checkpoint-9999.pth')) or overwrite:
                targets.append(shape_name)

    priority = ['tr_reg_088', 'tr_reg_090']
    prioritized = [p for p in priority if p in targets]
    others = [t for t in targets if t not in priority]
    return prioritized + others


def main(args):
    targets = get_targets_prioritized(overwrite=args.overwrite)
    if targets == []:
        print("No targets found, exiting.")
        exit(0)
    print(f"Processing {len(targets)} targets: {targets}")

    for target in targets:
        target_dir = Path(OUTPUT_DIR, target)

        working_dir = Path(str(Path(__file__).resolve()).split('/scripts')[0])
        data_path = Path(working_dir, SCAN_FAUST_DIR, f"{target}.ply")

        faust_landmarks = [412, 5891, 6593, 3323, 2119]
        corr = np.array(np.loadtxt(f'./data/SCAN-FAUST/full/corres/{target}.vts'))
        target_landmarks = corr[faust_landmarks]

        config = {
            "device": "cuda:0",
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
            "batch_size": 50000,
            "num_points_train": 50000,
            "learning_rate": 0.01,
            "distribution": "gaussian",
            "embedding_dim": 5,
            "embedding_type": "features_only",
            "features_type": "landmarks",
            "features_normalization": "diameter",
            "dists_path": "./data/SCAN-FAUST/full/dists/",
            "landmarks": target_landmarks.tolist(),
        }

        config_path = os.path.join(target_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)

        if args.external is True:
            command = [
                "python", "main.py",
                "--config", config_path,
                "--features_path", str(features_path),
                "--features_interpolation", str(500000),
            ]
        else:
            command = [
                "python", "main.py",
                "--config", config_path,
                "--features_interpolation", str('-1'),
            ]

        command_str = " ".join(command)
        print(f"Running command: {command_str}")

        try:
            subprocess.run(command, check=True)
            print(f"Successfully processed {target}")
        except subprocess.CalledProcessError as e:
            print(f"Error processing {target}: {e}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a flow on all FAUST scans PTs")
    parser.add_argument('--overwrite', action='store_true', help="Overwrite if an existing flow model \"checkpoint-9999.pth\" is found", default='False')
    parser.add_argument('--external', action='store_true', help="Use external precomputed features", default='False')

    args = parser.parse_args()

    print("Training flows on FAUST scan dataset with the following parameters:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print("-----------------------------------------------")

    main(args)
