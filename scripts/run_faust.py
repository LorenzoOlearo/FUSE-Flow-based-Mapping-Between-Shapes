import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import List


def get_targets(output_dir, overwrite) -> List[str]:
    targets = []

    for i in range(80, 100):
        target = f"tr_reg_{i:03d}"
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(Path(output_dir, target), exist_ok=True)

        if (
            os.path.exists(Path(output_dir, target, "checkpoint-9999.pth"))
            and overwrite == True
        ):
            targets.append(target)
        elif not os.path.exists(Path(output_dir, target, "checkpoint-9999.pth")):
            targets.append(target)

    return targets


def main(args):
    with open(args.config) as f:
        config = json.load(f)
    matching_config = config["matching_config"]["FAUST"]

    dataset_dir = Path(matching_config["dataset_path"])
    dists_path = matching_config["dists_path"]
    landmarks = matching_config["landmarks"]
    output_dir = Path(matching_config["flows_path"])

    if args.run_name is not None:
        output_dir = Path(matching_config["flows_path"]).parent / args.run_name

    os.makedirs(output_dir, exist_ok=True)
    targets = get_targets(output_dir, args.overwrite)
    if targets == []:
        print("No targets found, exiting.")
        exit(0)

    print(f"Processing {len(targets)} targets: {targets}")

    for target in targets:
        target_dir = Path(output_dir, target)
        os.makedirs(target_dir, exist_ok=True)

        working_dir = Path(str(Path(__file__).resolve()).split("/scripts")[0])
        data_path = Path(working_dir, dataset_dir, f"{target}.ply")
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
            "epochs": 20_000,
            "num_steps": 64,
            "method": args.method,
            "network": "MLP",
            "edm_preconditioning": True,
            "mlp_hidden_size": 256,
            "mlp_depth": 4,
            "mlp_num_frequencies": -1,
            "batch_size": 50_000,
            "num_points_train": 50_000,
            "learning_rate": 0.001,
            "distribution": "gaussian",
            "embedding_dim": len(landmarks),
            "embedding_type": "features_only",
            "features_type": "landmarks",
            "use_heat_method": False,
            "features_normalization": "diameter",
            "dists_path": dists_path,
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
        "--method",
        type=str,
        default="FM",
        help="Method to use to construct the flows: FM or Diffusion (DDIM)",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Override the output directory; saves to matching_config.FAUST.flows_path/../<run_name>/",
    )
    args = parser.parse_args()

    print("Training flows on FAUST dataset:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print("-----------------------------------------------")

    main(args)
