import subprocess
import os
import json
import numpy as np

def run_shapes():
    # Base directory for outputs
    output_base_dir = "faust_experiments/faust_wsk_plus_one_ldmk"
    
    # Create output directories if they don't exist
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Loop through shape indices from 80 to 99
    for i in range(80, 100):
        # Format shape name with leading zeros
        shape_name = f"tr_reg_{i:03d}"
        
        # Create specific output directory for this shape
        output_dir = f"{output_base_dir}/{shape_name}"
        log_dir = f"{output_base_dir}/{shape_name}"
        
        # Create directories if they don't exist
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # Build command with all parameters
        config = {
            "blr": 5e-7,
            "output_dir": output_dir,
            "log_dir": log_dir,
            "data_path": f"/home/ubuntu/giulio_vigano/FlowMatching4Matching/data/faust/test_set/{shape_name}.off",
            "epochs": 10000,
            "method": "FM",
            "network": "MLP_general",
            "batch_size": 10000,
            "num_points_train": 10000,
            "learning_rate": 0.01,
            "distribution": "Gaussian",
            "train": True,
            "embedding_dim": 11,
            "embedding": "features_only",
            "features_type": "wks_plus_ldmk",
            "device": "cuda:1",
            "landmarks": [412],
        }
        
        # Convert command list to string for printing
        config_path = os.path.join(output_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)

        command = [
            "python", "main.py",
            "--config", config_path
        ]
        checkpoint_path = os.path.join(output_dir, 'checkpoint-9999.pth')
        if os.path.exists(checkpoint_path):
            print(f"Skipping {shape_name} as checkpoint file exists.")
        else:
            # Convert command list to string for printing
            command_str = " ".join(command)
            print(f"Running command: {command_str}")

            # Execute the command
            try:
                subprocess.run(command, check=True)
                print(f"Successfully processed {shape_name}")
            except subprocess.CalledProcessError as e:
                print(f"Error processing {shape_name}: {e}")

if __name__ == "__main__":
    run_shapes()