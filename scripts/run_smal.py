import subprocess
import os
import numpy as np
import json


def run_shapes():
    # Base directory for outputs
    output_base_dir = "smal_experiments/smal_wks_only/"
    
    # Create output directories if they don't exist
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Loop through shape indices from 80 to 99
    for file in os.listdir("../datasets/meshes/SMAL_r/off/"):
        if file.endswith(".off"):
            shape_name = os.path.splitext(file)[0]
            # Create specific output directory for this shape
            output_dir = f"{output_base_dir}/{shape_name}"
            log_dir = f"{output_base_dir}/{shape_name}"
            
            # Create directories if they don't exist
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(log_dir, exist_ok=True)

            
            lm_smal=np.array([3162, 1931, 3731, 1399 ,1111, 1001])

            corr=np.array(np.loadtxt('../datasets/meshes/SMAL_r/corres/'+shape_name+'.vts'))-1
            lm = corr[lm_smal]


            # Build command with all parameters
            # Prepare config dictionary
            config = {
                "blr": 5e-7,
                "output_dir": output_dir,
                "log_dir": log_dir,
                "data_path": f"../datasets/meshes/SMAL_r/off/{shape_name}.off",
                "epochs": 10000,
                "num_step": 64,
                "method": "FM",
                "network": "MLP_general",
                "batch_size": 10000,
                "num_points_train": 10000,
                "learning_rate": 0.01,
                "distribution": "Gaussian",
                "train": True,
                "embedding_dim": 10,
                "embedding": "features_only",
                "features_type": "wks_plus_ldmk",
                "device": "cuda:1",
                "landmarks": lm.tolist(),
            }

            # Write config to a temporary file
            config_path = os.path.join(output_dir, "config.json")
            with open(config_path, "w") as f:
                json.dump(config, f, indent=4)

            # Build command to call main_smal.py with config
            command = [
                "python", "/home/ubuntu/FlowMatching4Matching/main.py",
                "--config", config_path
            ]
            # Check if the checkpoint file exists
            checkpoint_path = os.path.join(output_dir, 'checkpoint-9999.pth')
            if os.path.exists(checkpoint_path):
                print(f"Skipping {shape_name} as checkpoint file exist.")
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