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