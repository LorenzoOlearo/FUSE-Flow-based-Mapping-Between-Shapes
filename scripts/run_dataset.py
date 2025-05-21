import subprocess
import os
import json
import numpy as np
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run Flow Matching with configurations"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the configuration JSON file"
    )
    parser.add_argument(
        "--dry_run", action="store_true", help="Print commands without executing them"
    )
    return parser.parse_args()


def get_shape_files(config):
    dataset_config = config["dataset"]
    dataset_type = dataset_config.get("type", "range")

    if dataset_type == "range":
        # Generate filenames from a range of indices
        start = dataset_config.get("range_start", 0)
        end = dataset_config.get("range_end", 1)
        file_format = dataset_config.get("file_format", "{:d}")
        file_extension = dataset_config.get("file_extension", ".off")
        data_path = dataset_config.get("data_path", "./data/")

        files = []
        for i in range(start, end):
            shape_name = file_format.format(i)
            file_path = os.path.join(data_path, f"{shape_name}{file_extension}")
            files.append((shape_name, file_path))

        return files

    elif dataset_type == "directory":
        # Get all files in a directory
        directory = dataset_config.get("directory", "./data/")
        file_extension = dataset_config.get("file_extension", ".off")

        files = []
        for file in os.listdir(directory):
            if file.endswith(file_extension):
                shape_name = os.path.splitext(file)[0]
                file_path = os.path.join(directory, file)
                files.append((shape_name, file_path))

        return files

    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def get_landmarks(config, shape_name):
    dataset_config = config["dataset"]
    dataset_type = dataset_config.get("type", "range")

    if "landmarks" in dataset_config:
        # Use direct landmarks specification
        return dataset_config["landmarks"]

    elif "base_landmarks" in dataset_config and "correspondence_dir" in dataset_config:
        # Use landmarks with correspondence mapping
        base_landmarks = np.array(dataset_config["base_landmarks"])
        correspondence_dir = dataset_config["correspondence_dir"]
        correspondence_extension = dataset_config.get(
            "correspondence_extension", ".vts"
        )

        correspondence_file = os.path.join(
            correspondence_dir, f"{shape_name}{correspondence_extension}"
        )
        if os.path.exists(correspondence_file):
            corr = np.array(np.loadtxt(correspondence_file)).astype(np.int32)
            return corr[base_landmarks].tolist()
        else:
            print(
                f"Warning: Correspondence file {correspondence_file} not found. Using base landmarks."
            )
            return base_landmarks.tolist()

    else:
        # Default landmark if none specified
        return [0]


def run_shapes(config, dry_run=False):
    # Create output base directory
    output_base_dir = config["output"]["base_dir"]
    os.makedirs(output_base_dir, exist_ok=True)

    # Get all shape files to process
    shape_files = get_shape_files(config)
    print(f"Found {len(shape_files)} files to process")

    # Loop through each shape
    for shape_name, data_path in shape_files:
        # Create specific output directory for this shape
        output_dir = os.path.join(output_base_dir, shape_name)
        log_dir = os.path.join(output_base_dir, shape_name)

        # Create directories if they don't exist
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        # Get landmarks for this shape
        landmarks = get_landmarks(config, shape_name)

        # Build shape-specific configuration
        shape_config = {
            "blr": config["training"]["blr"],
            "output_dir": output_dir,
            "log_dir": log_dir,
            "data_path": data_path,
            "epochs": config["training"]["epochs"],
            "method": config["training"]["method"],
            "network": config["training"]["network"],
            "batch_size": config["training"]["batch_size"],
            "num_points_train": config["training"]["num_points_train"],
            "learning_rate": config["training"]["learning_rate"],
            "distribution": config["training"]["distribution"],
            "train": True,
            "embedding_dim": config["training"]["embedding_dim"],
            "embedding": config["training"]["embedding"],
            "features_type": config["training"]["features_type"],
            "device": config["training"]["device"],
            "landmarks": landmarks,
        }

        # Save configuration to file
        config_path = os.path.join(output_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(shape_config, f, indent=4)

        # Check if checkpoint exists
        checkpoint_name = config["execution"].get(
            "checkpoint_name", f"checkpoint-{config['training']['epochs'] - 1}.pth"
        )
        checkpoint_path = os.path.join(output_dir, checkpoint_name)

        if os.path.exists(checkpoint_path) and config["execution"].get(
            "skip_existing", True
        ):
            print(f"Skipping {shape_name} as checkpoint file exists.")
            continue

        # Build command
        command = ["python", "main.py", "--config", config_path]

        # Convert command list to string for printing
        command_str = " ".join(command)
        print(f"Running command: {command_str}")

        # Execute the command
        if not dry_run:
            try:
                subprocess.run(command, check=True)
                print(f"Successfully processed {shape_name}")
            except subprocess.CalledProcessError as e:
                print(f"Error processing {shape_name}: {e}")
        else:
            print("Dry run - command not executed")


if __name__ == "__main__":
    args = parse_arguments()

    # Load configuration file
    with open(args.config, "r") as f:
        config = json.load(f)

    # Run with the provided configuration
    run_shapes(config, args.dry_run)
