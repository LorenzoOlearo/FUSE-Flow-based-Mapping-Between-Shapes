import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import List

import numpy as np


def get_targets(output_dir, dataset_dir, overwrite) -> List[str]:
    targets = []

    for file in sorted(os.listdir(dataset_dir)):
        if not file.endswith(".off"):
            continue
        target = os.path.splitext(file)[0]
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
    mc = config["matching_config"]["TOPKIDS"]

    dataset_dir = Path(mc["dataset_path"])
    dists_path = mc["dists_path"]
    corr_path = mc["corr_path"]
    base_landmarks = mc["landmarks"]
    output_dir = Path(mc["flows_path"])

    if args.run_name is not None:
        output_dir = Path(mc["flows_path"]).parent / args.run_name

    os.makedirs(output_dir, exist_ok=True)
    targets = get_targets(output_dir, dataset_dir, args.overwrite)
    if targets == []:
        print("No targets found, exiting.")
        exit(0)

    print(f"Processing {len(targets)} targets: {targets}")

    for target in targets:
        target_dir = Path(output_dir, target)
        os.makedirs(target_dir, exist_ok=True)

        working_dir = Path(str(Path(__file__).resolve()).split("/scripts")[0])
        data_path = Path(working_dir, dataset_dir, f"{target}.off")

        # kid00 is the template — use landmarks directly.
        # All other shapes map template landmarks via their corr file.
        if target == "kid00":
            target_landmarks = base_landmarks
        else:
            # corr[kidNN_i] = kid00 vertex (kidNN→kid00 direction, 1-indexed → 0-indexed).
            # Find the kidNN vertex that maps to each template landmark via inverse lookup.
            corr = np.array(np.loadtxt(f"{corr_path}/{target}.vts")) - 1
            target_landmarks = [int(np.where(corr == L)[0][0]) for L in base_landmarks]

        train_config = {
            "device": "cuda:0",
            "blr": 5e-7,
            "output_dir": str(target_dir),
            "log_dir": str(target_dir),
            "data_path": str(data_path),
            "train": True,
            "inference": True,
            "epochs": 20_000,
            "num_steps": 64,
            "method": args.method,
            "network": "MLP",
            "edm_preconditioning": False,
            "mlp_hidden_size": 256,
            "mlp_depth": 4,
            "mlp_num_frequencies": 6,
            "batch_size": 10_000,
            "num_points_train": 10_000,
            "learning_rate": 0.001,
            "lr_scheduler": "none",
            "distribution": "gaussian",
            "embedding_dim": len(base_landmarks),
            "embedding_type": "features_only",
            "features_type": "landmarks",
            "use_heat_method": False,
            "features_normalization": "diameter",
            "dists_path": dists_path,
            "landmarks": target_landmarks,
            "warmup_epochs": 100,
            "min_lr": 1e-4,
        }

        config_path = os.path.join(target_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(train_config, f, indent=4)

        command = [
            "python",
            "train.py",
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
    parser = argparse.ArgumentParser(description="Train a flow on all TOPKIDS meshes")
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
        "--method",
        type=str,
        default="FM",
        help="Method to use to construct the flows: FM or Diffusion (DDIM)",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Override the output directory; saves to matching_config.TOPKIDS.flows_path/../<run_name>/",
    )
    args = parser.parse_args()

    print("Training flows on TOPKIDS dataset:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print("-----------------------------------------------")

    main(args)
