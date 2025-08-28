# FlowMatching4Matching

This repository provides a clean and modular implementation of **Geometry Distributions** and a novel variant inspired by the **Flow Matching** approach. Our proposed representation is flexible and can leverage any input features—particularly distances between landmarks—to learn effective embeddings for shape matching tasks.

---

## 🚀 Features

- **Modular implementation** of two key approaches:
  - **GeomDist**: Uses Euclidean Distance Matrix (EDM) trajectories.
  - **FlowMatching (FM)**: Learns flow-based trajectories for shape matching.
- **Easily configurable** via JSON files.
- **Training and inference logging**, checkpoint saving, and feature visualization.
- **Ready-to-use scripts** for dataset-based training and evaluation.

---

## 🧠 Method Overview

We represent geometric structures by transforming Gaussian distributions into a feature space defined by distances between landmarks. Two modeling options are available:

- **GeomDist**: Uses deterministic geometry-aware transformations.
- **FlowMatching**: Leverages neural ODEs for probabilistic flow matching between distributions.

---

## 🛠️ Getting Started

### 1. Train a Model

Run the following command to start training:

```bash
python main.py --config configs/model/config.json
```

You can customize:
- The features used (e.g., distances, angles).
- The model type: choose between `FM` (Flow Matching) or `GeomDist`.

Training will output:
- Model checkpoints that transform Gaussian noise into learned features.
- Training features and inferred features.
- Logs detailing the training process.

---

### 2. Run on a Dataset

To train on a specific dataset:

```bash
python scripts/run_dataset.py --config configs/matching/faust_ldmk.json
```

---

### 3. Evaluate Matching Performance

After training, you can evaluate the matching results using:

```bash
python scripts/matching_scripts.py --config configs/matching/faust_ldmk.json
```

---

### Utilities

#### **SDF Training**

Given a mesh file, you can train a neural Signed Distance Function (SDF) model.
Optionally, if landmark indices are available, the script can also convert mesh
landmark coordinates to the SDF volume grid.

To train an SDF model from a mesh, run:

```bash
python scripts/train_SDF.py --mesh_path /path/to/mesh/tr_reg_082.ply --eval
```

The `--eval` flag will evaluate the SDF model after training and converting the
mesh landmarks to the SDF volume grid.

Output:

- The trained SDF model will be saved as ```<FILENAME>-SDF.pth``` in the
```./out/SDFs directory```.

- If landmark indices are provided, a corresponding
```<FILENAME>-landmark-voxels.ply``` file will also be saved, containing
landmark coordinates transformed into the SDF volume grid.


Alternatively, you can train an SDF model for the entire dataset by running:

```bash
python scripts/train_SDF.py --mesh_path /path/to/dataset --eval [--test / --all]
````

By providing the `--test` flag, the script will train an SDF model only for the
80 to 99 samples in the FAUST dataset. The `--all` flag will train SDFs for all
meshes in the dataset.


#### SDFs Feature Extraction
We use landmark distances as features, in the case of SDFs, the distances are
computed by constructing a shortest path between the landmarks in the SDF
volume grid with Dijkstra's algorithm.

To extract features from the SDFs, run:

```bash
python scripts/extract_features_SDF.py --target tr_reg_097 --num_points 500000 --mesh_path ./data/MPI-FAUST/training/registrations/tr_reg_097.ply
```

Or, to extract feature for multiple SDFs in the FAUST dataset, run:

```bash
python scripts/extract_features_SDF.py --test --mesh_folder ./data/MPI-FAUST/training/registrations --additional_plots --num_points 500000
```

where `--test` will extract features for the 80 to 99 samples in the FAUST
dataset, and `--additional_plots` will save additional plots to visualize the
extracted features.

#### Training a Flow on the computed features
Once you have trained the SDFs and extracted the features, you can create a
Flow Matching representation that will be used to solve the correspondence
problem.

To train a Flow Matching model on the extracted features, run:

```bash
python scripts/run_faust_SDFs.py --test --overwrite
```

where `--test` will run the training on the 80 to 99 samples in the FAUST and
`--overwrite` will overwrite the existing model checkpoints.

#### Let's Match!
Having trained the Flow Matching model, we can now match SDFs using these
representations.

TODO

## 📁 Repository Structure

```
configs/
  model/                 # Model training configurations
  matching/              # Dataset-specific configs for training + evaluation
scripts/
  run_dataset.py         # Runs training on a dataset
  matching_scripts.py    # Evaluates performance post-training
main.py                  # Core training script
```

---

## 📌 Notes

- Ensure your dataset paths and features are correctly set in the config files.
- The method supports generalization across datasets with different types of landmark representations.
