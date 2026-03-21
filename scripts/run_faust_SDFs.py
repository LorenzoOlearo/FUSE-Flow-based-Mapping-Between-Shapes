import argparse
import json
import os
import re
import subprocess
from pathlib import Path
from typing import List


def get_targets(output_dir, sdf_dir, overwrite, test) -> List[str]:
    """Process all SDFs on which mesh vertex distances have been computed"""
    if overwrite and test:
        targets = [
            f.name
            for f in sdf_dir.iterdir()
            if f.is_dir()
            and any(80 <= int(num) <= 99 for num in re.findall(r"\d+", f.name))
            and any(child.suffix == ".pth" for child in f.iterdir())
            and any(
                child.name.endswith("sdf-dijkstra-surface-points.txt")
                for child in f.iterdir()
            )
        ]
    elif overwrite == True:
        targets = [
            f.name
            for f in sdf_dir.iterdir()
            if f.is_dir()
            and any(child.suffix == ".pth" for child in f.iterdir())
            and any(
                child.name.endswith("sdf-dijkstra-surface-points.txt")
                for child in f.iterdir()
            )
        ]
    elif test == True:
        targets = [
            f.name
            for f in sdf_dir.iterdir()
            if f.is_dir()
            and any(80 <= int(num) <= 99 for num in re.findall(r"\d+", f.name))
            and any(child.suffix == ".pth" for child in f.iterdir())
            and any(
                child.name.endswith("sdf-dijkstra-surface-points.txt")
                for child in f.iterdir()
            )
        ]
    else:
        targets = []
        for f in sdf_dir.iterdir():
            if not f.is_dir():
                continue
            has_pth = any(child.suffix == ".pth" for child in f.iterdir())
            has_sdf = any(
                child.name.endswith("sdf-dijkstra-surface-points.txt")
                for child in f.iterdir()
            )
            if not (has_pth and has_sdf):
                continue
            checkpoint = Path(output_dir, f.name, "checkpoint-9999.pth")
            if checkpoint.exists():
                continue
            targets.append(f.name)
    return targets


def main(args):
    with open(args.config) as f:
        config = json.load(f)
    matching_config = config["matching_config"]["FAUST"]

    dataset_dir = Path(matching_config["dataset_path"])
    sdf_dir = Path(matching_config["SDFs_path"])
    landmarks = matching_config["landmarks"]
    output_dir = Path(matching_config["flows_SDFs_path"])

    if args.run_name is not None:
        output_dir = Path(matching_config["flows_SDFs_path"]).parent / args.run_name

    os.makedirs(output_dir, exist_ok=True)
    targets = get_targets(output_dir, sdf_dir, args.overwrite, args.test)
    if targets == []:
        print("No targets found, exiting.")
        exit(0)

    print(f"Processing {len(targets)} targets: {targets}")

    for target in targets:
        target_dir = Path(output_dir, target)
        os.makedirs(target_dir, exist_ok=True)

        working_dir = Path(str(Path(__file__).resolve()).split("/scripts")[0])
        data_path = Path(working_dir, dataset_dir, f"{target}.ply")

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
            "embedding_dim": len(landmarks),
            "embedding_type": "features_only",
            "features_type": "landmarks",
            "features_normalization": "none",
            "landmarks": landmarks,
        }

        config_path = os.path.join(target_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)

        command = [
            "python",
            "main.py",
            "--config",
            config_path,
            "--features_path",
            f"{sdf_dir}/{target}/{target}-geodesics-normalized-diameter.txt",
            "--vertex_features_path",
            f"{sdf_dir}/{target}/{target}-vertex-geodesics-normalized-diameter.txt",
        ]

        command_str = " ".join(command)
        print(f"Running command: {command_str}")

        try:
            subprocess.run(command, check=True)
            print(f"Successfully processed {target}")
        except subprocess.CalledProcessError as e:
            print(f"Error processing {target}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a flow on all FAUST SDFs features"
    )
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
        "--test",
        action="store_true",
        help="Test mode, only process targets that are not in the range 80-99",
        default="False",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Override the output directory; saves to matching_config.FAUST.flows_SDFs_path/../<run_name>/",
    )

    args = parser.parse_args()

    print("Training flows on FAUST SDFs features:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print("-----------------------------------------------")

    main(args)
