# FUSE: A Flow-based Mapping Between Shapes

__Authors__: Lorenzo Olearo*, Giulio Viganò*, Daniele Baieri, Filippo Maggioli, Simone Melzi

![](assets/cover.png)

We introduce a novel neural representation for maps between 3D shapes based on
flow-matching models, which is computationally efficient and supports
cross-representation shape matching without large-scale training or data-driven
procedures. 3D shapes are represented as the probability distribution induced
by a continuous and invertible flow mapping from a fixed anchor distribution.
Given a source and a target shape, the composition of the inverse flow (source
to anchor) with the forward flow (anchor to target), we map points between the
two surfaces. By encoding the shapes with a pointwise task-tailored embedding,
this construction provides an invertible and modality-agnostic representation
of maps between shapes across point clouds, meshes, signed distance fields
(SDFs), and volumetric data. The resulting representation consistently achieves
high coverage and accuracy across diverse benchmarks and challenging settings
in shape matching. Beyond shape matching, our framework shows promising results
in other tasks, including UV mapping and registration of raw point cloud scans
of human bodies.

- [![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2511.13431) - **FUSE: A Flow-based Mapping Between Shapes**

**Citation**:
```
@misc{olearo2025fuse,
      title={FUSE: A Flow-based Mapping Between Shapes}, 
      author={Lorenzo Olearo and Giulio Viganò and Daniele Baieri and Filippo Maggioli and Simone Melzi},
      year={2025},
      eprint={2511.13431},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2511.13431}, 
}
```

---

## Table of Contents

1. [Repository Structure](#repository-structure)
2. [Configuration](#configuration)
3. [Training a Flow for a single shape (main.py)](#training-a-single-shape-mainpy)
4. [Training an Entire Dataset (scripts/run\_\*.py)](#training-an-entire-dataset-scriptsrun_py)
5. [Neural SDF Training (scripts/train\_SDF.py)](#neural-sdf-training-scriptstrain_sdfpy)
6. [Shape Matching Evaluation (matching\_experiments.py)](#shape-matching-evaluation-matching_experimentspy)
7. [Configuration and Dataset Paths](#configuration-and-dataset-paths)

---

## Repository Structure

```
FlowMatching4Matching/
├── main.py                        # Core single-shape training and inference script
├── matching.py                    # Perform shape matching evaluation across specified datasets and methods
│
├── matching/                      # Matching pipeline modules
│   ├── data_structures.py         # DataPath, Element, MatchingResult dataclasses
│   ├── targets.py                 # Per-dataset target discovery (get_targets_*)
│   ├── element_processing.py      # Load shapes, features, and flow models per element
│   ├── methods.py                 # builds the method dispatch table
│   ├── evaluation.py              # run_matching_methods, per-dataset metric evaluation (SHREC19/20, FAUST scan, default), log_results
│   ├── visualization.py           # plotly/matplotlib result visualizations
│   ├── pipeline.py                # matching pipeline for each source-target pair
│   └── p2p/                       # Point-to-point algorithm implementations
│       ├── flow.py                # Flow composition methods (standard, hungarian, lapjv, zoomout, in-gauss)
│       ├── knn.py                 # KNN matching with optional ZoomOut refinement
│       ├── fmaps.py               # Functional maps methods (standard, WKS, zoomout, neural zoomout)
│       ├── assignment.py          # Linear assignment methods (Hungarian, LAPJV)
│       ├── ot.py                  # Optimal transport (Sinkhorn)
│       └── ndp.py                 # Neural Deformation Pyramid (landmark-guided, SDF)
│
├── model/
│   ├── models.py                  # FMCond (Flow Matching) and EDMPrecond (from Geometry Distributions)
│   ├── networks.py                # MLP backbone with sinusoidal/Fourier embeddings
│   └── losses.py                  # ChamferLoss, HausdorffLoss
│
├── util/
│   ├── mesh_utils.py              # Geodesic computation, feature extraction, mesh sampling
│   ├── metrics.py                 # Geodesic error, coverage, Dirichlet energy
│   ├── dataset_utils.py           # Dataset-specific helpers and SHREC20 landmark loading
│   ├── misc.py                    # MetricLogger, checkpoint save/load, distributed utils
│   ├── plot.py                    # Plotly 3D visualizations
│   └── train_utils.py             # Device initialization, seed, logging setup
│
├── scripts/
│   ├── run_faust.py               # Train flows on FAUST dataset (meshes)
│   ├── run_faust_r.py             # Train flows on FAUST-R (remeshed)
│   ├── run_faust_SDFs.py          # Train flows on FAUST SDF features
│   ├── run_smal.py                # Train flows on SMAL dataset
│   ├── run_smal_SDFs.py           # Train flows on SMAL SDF features
│   ├── run_shrec19.py             # Train flows on SHREC19 dataset
│   ├── run_shrec20.py             # Train flows on SHREC20 dataset
│   ├── run_kinect.py              # Train flows on KINECT point clouds
│   ├── run_surreal.py             # Train flows on SURREAL dataset
│   ├── run_tosca.py               # Train flows on TOSCA dataset
│   ├── run_scan_faust.py          # Train flows on SCAN-FAUST (raw scans)
│   ├── run_dataset.py             # Generic dataset training script
│   ├── train_SDF.py               # Train neural SDFs (IGR) on mesh data
│   ├── extract_features_SDF.py    # Extract landmark-distance features from SDF volumes
│   ├── kinect_on_smplx.py         # SMPLX template to KINECT skinning experiment
│   └── sphere_parametrization.py  # Sphere parametrization experiments
│
├── configs/
│   └── matching/                  # JSON configs for matching evaluation
│
├── data/                          # Dataset directories (see Dataset Paths)
└── out/                           # All outputs: flows, SDFs, matching results
    ├── flows/                     # Per-shape trained FM/DDIM models
    ├── SDFs/                      # Trained neural SDF models and extracted features
    └── matching/                  # Matching evaluation CSVs and visualizations
```

The core training code is available in [`main.py`](main.py), which is designed
to train either a flow or diffusion model (DDIM) on a single shape. This
implementation builds upon the work of Zhang et al. (2025), *Geometry
Distributions*, to facilitate a comparison between Flow Matching and DDIM
models.

- **Code Repository:** [1zb/GeomDist (GitHub)](https://github.com/1zb/GeomDist)
- **Paper:** [Geometry Distributions (ICCV 2025)](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhang_Geometry_Distributions_ICCV_2025_paper.pdf)

---

## Configuration

All training is controlled by JSON config files. The key fields are:

| Field | Description | Example values |
|---|---|---|
| `method` | Training method | `"FM"`, `"DDIM"` |
| `network` | Network backbone | `"MLP"`, `"GEOMDIST"` |
| `edm_preconditioning` | Apply EDM-style sigma preconditioning inside FM (see note below) | `false` (default), `true` |
| `embedding_dim` | feature dimension, if geodesics are used, this is the number of landmarks | `5`, `6`, `20` |
| `embedding_type` | How features are presented to the model | `"features_only"`, `"xyz"`, `"features_and_xyz"` |
| `features_type` | What features to compute | `"landmarks"`, `"wks", "wks_landmarks", "hks", hks_landmarks"` |
| `features_normalization` | Normalization strategy | `"diameter"`, `"0_1_independent"`, `"mean_var"`, `"none"` |
| `distribution` | Initial distribution for the flow | `"gaussian"`, `"sphere"`, `"cube"`, `"sphere_surface"` |
| `landmarks` | Vertex indices of landmark points | `[412, 5891, 6593, 3323, 2119]` |
| `data_path` | Path to the input mesh | `"./data/MPI-FAUST/..."` |
| `output_dir` | Where to save checkpoints and features | `"./out/flows/faust/..."` |
| `epochs` | Number of training epochs | `10000` |
| `batch_size` | Points sampled per gradient step | `10000` |
| `learning_rate` | Optimizer learning rate | `0.0001` |
| `lr_scheduler` | LR scheduler: `"cosine"` (warmup + half-cycle cosine decay), `"plateau"` (`ReduceLROnPlateau`), or `"none"` (constant LR) | `"cosine"` |
| `lr_patience` | `(plateau)` epochs with no improvement before reducing LR | `200` |
| `lr_factor` | `(plateau)` multiplicative LR reduction factor | `0.5` |
| `mlp_hidden_size` | Hidden dimension of the MLP | `256` |
| `mlp_depth` | Number of MLP layers | `4` |

Config files for matching evaluation live in `configs/matching/`. Per-shape
training configs are generated automatically by the `scripts/run_*.py` scripts
and saved alongside the model checkpoints.

> **Note on `edm_preconditioning`.**
> By default, FM and DDIM differ not only in their training objectives and
> sampling procedures, but also in how the shared network is called. DDIM
> wraps every forward pass with the EDM sigma-preconditioning scheme from
> by Zhang et al. (2025), *Geometry Distributions*: the input is scaled by
> `c_in = 1/√(σ_data² + t²)`, the time step is transformed to
> `c_noise = log(t)/4`, and the raw network output is blended with a skip
> connection `D(x,t) = c_skip·x + c_out·F(c_in·x, c_noise)`.
> Setting `edm_preconditioning: true` applies the identical transformation
> inside `FMCond.forward()`, making the two methods fully comparable: the only
> remaining differences are the training objective (velocity matching vs.
> denoising score matching), the time schedule (uniform `t ∈ [0,1]` vs.
> log-normal σ), and the ODE solver used at inference. This preconditioning is
> applied consistently during both training and inference — the ODE solver
> always calls `FMCond.forward()` directly so the scaling is never bypassed.

---

## Training a Flow for a Single Shape (`main.py`)

`main.py` trains a Flow or DDIM model on a single mesh. It handles feature
computation, optional interpolation over sampled points, normalization, and the
training loop.

### Basic usage

```bash
python main.py --config config.json
```

CLI arguments override config file values. The most commonly overridden arguments:

Run training and inference on a FAUST mesh, with feature interpolation:
```bash
python main.py \
    --config config.json \
    --features_interpolation 100000 \
    --learning_rate 0.01
```

Run on a SMAL mesh with a pre-saved config:
```bash
python main.py \
    --config out/flows/smal/smal-diameter-norm/horse_04/config.json \
    --features_interpolation 50000 \
    --learning_rate 0.01
```

Load externally precomputed features (e.g., FMNet descriptors):
```bash
python main.py \
    --config /path/to/horse_04/config.json \
    --vertex_features_path /path/to/horse_04.npy \
    --features_interpolation 50000
```

Treat the mesh as a point cloud (remove all faces):
```bash
python main.py --config config.json --pt
```

### Training pipeline

1. Load the mesh (or point cloud) from `data_path`.
2. Compute or load the features for all vertices (or points).
3. Optionally interpolate features to `features_interpolation` randomly sampled
   surface points.
4. Normalize features according to `features_normalization` (e.g., divide by
   mesh diameter).
5. Train the FM or DDIM model for `epochs` epochs, saving the best checkpoint.

### Key arguments

| Argument | Description |
|---|---|
| `--config` | Path to JSON config file |
| `--train` | Run training (implied when config is given with training fields) |
| `--inference` | Run inference with the saved best checkpoint |
| `--method` | `FM` or `DDIM` |
| `--features_type` | Feature type to compute (`landmarks`, `wks`, etc.) |
| `--features_normalization` | Normalization method |
| `--features_interpolation` | Number of points to interpolate features onto (`-1` to disable) |
| `--vertex_features_path` | Path to externally precomputed per-vertex features (`.npy` or `.txt`) |
| `--distribution` | Prior distribution (`gaussian`, `sphere`, `cube`, `sphere_surface`) |
| `--embedding_dim` | Feature dimension (= number of landmarks) |
| `--epochs` | Number of training epochs |
| `--learning_rate` | Learning rate |
| `--lr_scheduler` | `cosine` (default), `plateau` (`ReduceLROnPlateau`), or `none` (constant LR) |
| `--lr_patience` | `(plateau)` epochs before LR reduction (default: 200) |
| `--lr_factor` | `(plateau)` LR reduction factor (default: 0.5) |
| `--mlp_hidden_size` | MLP hidden dimension |
| `--mlp_depth` | Number of MLP layers |
| `--pt` | Treat the input as a point cloud (removes all faces) |
| `--use_heat_method` | Use the heat method for geodesics instead of Dijkstra |
| `--edm_preconditioning` | Apply EDM-style sigma preconditioning inside `FMCond` (makes FM and DDIM fully comparable; see Configuration note) |
| `--output_dir` | Directory for checkpoints and feature files |
| `--resume` | Path to a checkpoint to resume training from |

### Outputs

Saved in `output_dir`:
- `checkpoint-best.pth` — best model checkpoint by training loss
- `checkpoint-<epoch>.pth` — final epoch checkpoint
- `vertex-geodesics.txt` — raw per-vertex features
- `vertex-geodesics-interpolated.txt` — features interpolated over sampled points
- `vertex-geodesics-vnorm.txt` — normalized per-vertex features
- `vertex-geodesics-interpolated-vnorm.txt` — normalized interpolated features
- `loss.png` — training loss curve

---

## Training an Entire Dataset (`scripts/run_*.py`)

Each `run_<DATASET>.py` script iterates over all shapes in a dataset, generates
a per-shape `config.json`, and calls `main.py` as a subprocess. This is the
standard way to produce trained flows for an entire benchmark.

### Available scripts

| Script | Dataset |
|---|---|
| `scripts/run_faust.py` | FAUST (meshes, registrations) |
| `scripts/run_faust_r.py` | FAUST-R (remeshed) |
| `scripts/run_faust_SDFs.py` | FAUST (SDF feature space) |
| `scripts/run_smal.py` | SMAL (animal meshes) |
| `scripts/run_smal_SDFs.py` | SMAL (SDF feature space) |
| `scripts/run_shrec19.py` | SHREC19 |
| `scripts/run_shrec20.py` | SHREC20 |
| `scripts/run_kinect.py` | KINECT (point clouds) |
| `scripts/run_surreal.py` | SURREAL |
| `scripts/run_tosca.py` | TOSCA |
| `scripts/run_scan_faust.py` | SCAN-FAUST (raw human scans) |

### Arguments

| Flag | Description |
|---|---|
| `--config` | **(required)** Path to the matching config JSON — the same file used by `matching.py`. All dataset paths, dists paths, correspondence paths, and output paths are read from `matching_config.<DATASET>.*` |
| `--overwrite` | Retrain shapes that already have a saved checkpoint |
| `--external` | Load precomputed features instead of computing them from scratch |
| `--method` | Override the training method (`FM` or `DDIM`) |
| `--run_name` | Override the output directory; saves to the parent of the config's `flows_path` (or `scan_flows_path` / `flows_SDFs_path`) joined with `<run_name>` |

Each script reads all paths from the `matching_config` block of the config file:

| Path read from config | Used for |
|---|---|
| `<DATASET>.dataset_path` | Mesh directory |
| `<DATASET>.dists_path` | Geodesic distance cache |
| `<DATASET>.flows_path` | Output directory for trained flows |
| `<DATASET>.corr_path` | Correspondence files (FAUST_R, SMAL, KINECT, SHREC19) |
| `<DATASET>.landmarks` | Base landmark indices |
| `SHREC20.common_landmarks_path` | Per-shape landmark CSV |
| `FAUST.scan_*` | SCAN-FAUST paths (`run_scan_faust.py`) |
| `FAUST.SDFs_path`, `FAUST.flows_SDFs_path` | SDF pipeline (`run_faust_SDFs.py`) |
| `SMAL.flows_SDFs_path` | SMAL SDF flows (`run_smal_SDFs.py`) |

### Example usage

Train flows on all FAUST test shapes (tr_reg_080 to tr_reg_099):
```bash
python scripts/run_faust.py --config config.json
```

Train with a custom output directory:
```bash
python scripts/run_faust.py --config config.json --run_name my-experiment
```

Load externally precomputed features (e.g., FMNet, Diff3D):
```bash
python scripts/run_faust.py --config config.json --external
```

Train flows on SMAL dataset:
```bash
python scripts/run_smal.py --config config.json
```

Train flows on SHREC20 shapes using the DDIM trajectory instead of FM:
```bash
python scripts/run_shrec20.py --config config.json --method DDIM
```

Train on KINECT point cloud dataset:
```bash
python scripts/run_kinect.py --config config.json
```

Each script saves one subdirectory per shape under the `flows_path` specified
in the config, containing the model checkpoint and the feature files used by
`matching.py`.

---

## Neural SDF Training (`scripts/train_SDF.py`)

Given a mesh file, you can train a neural Signed Distance Function (SDF) model
using Implicit Geometric Regularization (IGR). Optionally, if landmark indices
are available, the script can also convert mesh landmark coordinates to the SDF
volume grid.

The neural SDF is an 8-layer MLP with Softplus activations, Fourier positional
encoding, skip connections, and geometric initialization. The training loss
combines a zero-level-set term (SDF is zero on the surface), an eikonal term
(gradient norm equals 1 everywhere), a normals alignment term, and a
free-space penalty to prevent the SDF from collapsing to zero.

### Training a neural SDF

To train an SDF model on a single mesh:

```bash
python scripts/train_SDF.py --mesh_path /path/to/mesh/tr_reg_082.ply --eval
```

The `--eval` flag evaluates the model after training: it extracts the
reconstructed mesh via marching cubes and, if landmark indices are configured,
maps the mesh landmark coordinates into the SDF volume grid.

To train on an entire dataset:

```bash
python scripts/train_SDF.py --mesh_folder /path/to/dataset --all --eval [--test]
```

The `--test` flag restricts training to shapes `tr_reg_080` through
`tr_reg_099` in the FAUST dataset. The `--all` flag processes every mesh in
`--mesh_folder`.

### Optional arguments

| Argument | Description |
|---|---|
| `--mesh_path` | Path to a single mesh file |
| `--mesh_folder` | Path to a folder of meshes (use with `--all`) |
| `--all` | Train on all meshes in `--mesh_folder` |
| `--eval` | After training, extract the mesh via marching cubes and map landmarks to voxel coordinates |
| `--test` | Restrict `--all` to FAUST test shapes (`tr_reg_080` to `tr_reg_099`) |
| `--smal` | Filter `--all` to SMAL animal species (`cougar`, `hippo`, `horse`) |

### Outputs

Saved in `out/SDFs/<mesh_name>/`:
- `<mesh_name>-SDF.pth` — trained SDF network weights
- `SDF-<mesh_name>.html` / `.png` — marching cubes mesh visualization
- `<mesh_name>-landmarks-voxels.npy` — landmark coordinates mapped to voxel indices
- `<mesh_name>-landmarks-vertices.npy` — original landmark vertex coordinates

### SDF feature extraction

Features for SDF-based matching are landmark distances computed by running
Dijkstra's algorithm on the SDF voxel grid, constructing shortest paths between
each landmark and all other grid points. Note that the landmarks used for the
geodesic embeddings are projected onto the SDF surface and mapped into their
corresponding voxel coordinates.

To extract features for a single shape:
```bash
python scripts/extract_features_SDF.py \
    --target tr_reg_097 \
    --num_points 500000 \
    --mesh_path ./data/MPI-FAUST/training/registrations/tr_reg_097.ply
```

To extract features for the FAUST test set:
```bash
python scripts/extract_features_SDF.py \
    --test \
    --mesh_folder ./data/MPI-FAUST/training/registrations \
    --num_points 500000 \
    --additional_plots
```

The `--additional_plots` flag saves extra visualizations of the reconstructed
SDF surface and the points sampled over it, the voxelized zero-level set, the
features computeted on the SDF grid, the features remapped back to the sampled
points and the projection of the mesh vertices onto the SDF surface.

### Training a flow on the extracted SDF features

Once SDFs are trained and features are extracted, train the Flow Matching
models that will be used for the later shape matching evaluation:

```bash
python scripts/run_faust_SDFs.py --test --overwrite
```

The `--test` flag processes shapes `tr_reg_080` through `tr_reg_099`;
`--overwrite` retrains shapes that already have a saved checkpoint.

---

## Shape Matching Evaluation (`matching.py`)

This script loads trained flow models for each shape in a dataset, runs all
selected matching methods on every source-target pair, and reports geodesic
error, Euclidean error, Dirichlet energy, and coverage.

### Usage

FAUST mesh-to-mesh matching (all pairs):
```bash
python matching.py \
    --config config.json \
    --faust \
    --source_rep mesh \
    --target_rep mesh \
    --mesh_baseline \
    --matching_methods fast \
    --matching_run_name my_experiment
```

Match a specific pair (FAUST: tr_reg_080 -> tr_reg_081):
```bash
python matching.py \
    --config config.json \
    --faust \
    --source_rep mesh --target_rep mesh \
    --mesh_baseline \
    --matching_methods fast \
    --source_shape tr_reg_080 \
    --target_shape tr_reg_081 \
    --matching_run_name DEBUG \
    --plot_png --plot_html
```

SMAL mesh-to-mesh matching:
```bash
python matching.py \
    --config config.json \
    --smal \
    --source_rep mesh --target_rep mesh \
    --mesh_baseline \
    --matching_methods fast \
    --matching_run_name DEBUG
```

SHREC20 cross-category matching (cow -> camel_a):
```bash
python matching.py \
    --config config.json \
    --shrec20 \
    --source_rep mesh --target_rep mesh \
    --mesh_baseline \
    --matching_methods fast \
    --source_shape cow --target_shape camel_a \
    --matching_run_name DEBUG \
    --plot_png --plot_html
```

KINECT point cloud matching:
```bash
python matching.py \
    --config config.json \
    --kinect \
    --source_rep pt --target_rep pt \
    --mesh_baseline \
    --matching_methods fast \
    --matching_run_name DEBUG
```

FAUST SDF-to-SDF matching:
```bash
python matching.py \
    --config config.json \
    --faust \
    --source_rep sdf --target_rep sdf \
    --mesh_baseline \
    --matching_methods sdf \
    --matching_run_name DEBUG
```

SMPLX template skinning onto KINECT point clouds:
```bash
python matching.py \
    --config config.json \
    --kinect --pt_skinning \
    --source_rep mesh --target_rep pt \
    --matching_methods fast \
    --matching_run_name DEBUG
```

Evaluate identity (same shape, used to evaluate the inversion error of the flow):
```bash
python matching.py \
    --config config.json \
    --faust \
    --source_rep mesh --target_rep mesh \
    --mesh_baseline \
    --matching_methods fast \
    --same \
    --matching_run_name DEBUG
```

### Dataset flags

| Flag | Dataset |
|---|---|
| `--faust` | FAUST (template meshes) |
| `--faust_r` | FAUST-R (remeshed) |
| `--smal` | SMAL (animal meshes) |
| `--shrec19` | SHREC19 |
| `--shrec20` | SHREC20 |
| `--kinect` | KINECT (point clouds) |
| `--surreal` | SURREAL |
| `--tosca` | TOSCA |

### Shape representations

The `--source_rep` and `--target_rep` flags control the representation for each
side of the pair:

| Value | Description |
|---|---|
| `mesh` | Triangle mesh; features computed from geodesics on the mesh surface |
| `pt` | Point cloud; geodesics computed without connectivity |
| `sdf` | Neural SDF; features computed via shortest paths through the SDF voxel grid |

### Matching methods

Methods are grouped by the `--matching_methods` flag:

| Group | Included methods |
|---|---|
| `fast` | `KNN`, `FUSE` |
| `all` | `KNN`, `OT`, `FMaps`, `FMaps-zoomout`, `FMaps-neural-zoomout`, `NDP-landmarks`, `NDP-wks`, `FUSE`, `FUSE-ANCHOR`, `FUSE-zoomout`, `FUSE-neural-zoomout` |
| `baselines` | `KNN`, `OT`, `FUSE`, `FMaps`, `FMaps-zoomout`, `FMaps-neural-zoomout`, `FUSE-ANCHOR`, `NDP-landmarks`, `NDP-wks` |
| `baselines-no-zoomout` | `KNN`, `OT`, `FUSE`, `FMaps`, `FUSE-ANCHOR`, `NDP-landmarks`, `NDP-wks` |
| `sdf` | `KNN`, `OT`, `FUSE`, `NDP-SDF`, `FUSE-ANCHOR` |
| `zoomout` | `FMaps`, `FMaps-zoomout`, `FMaps-neural-zoomout` |
| `la` | `KNN`, `FUSE`, `hungarian`, `lapjv`, `FUSE-hungarian`, `FUSE-lapjv` |

Individual methods:

| Method | Description |
|---|---|
| `KNN` | Nearest-neighbor matching in feature space |
| `OT` | Optimal transport matching in feature space |
| `FUSE` | Flow composition: invert source flow, apply target flow, then KNN |
| `FUSE-ANCHOR` | KNN matching after mapping both shapes to the anchor distribution via their inverse flows |
| `FMaps` | Functional maps (spectral) |
| `FMaps-zoomout` | Functional maps with ZoomOut refinement |
| `FMaps-neural-zoomout` | Functional maps with neural ZoomOut refinement |
| `NDP-wks` | Functional maps using Wave Kernel Signature descriptors |
| `NDP-landmarks` | Non-rigid deformation with landmark constraints |
| `NDP-SDF` | Non-rigid deformation using SDF-projected vertex positions |
| `FUSE-zoomout` | Flow composition followed by ZoomOut refinement |
| `FUSE-neural-zoomout` | Flow composition followed by neural ZoomOut refinement |
| `hungarian` | Hungarian algorithm for optimal assignment in feature space |
| `lapjv` | Jonker-Volgenant algorithm for linear assignment in feature space |
| `FUSE-hungarian` | Flow composition followed by Hungarian assignment |
| `FUSE-lapjv` | Flow composition followed by Jonker-Volgenant assignment |

### Key arguments

| Argument | Description |
|---|---|
| `--config` | Path to the JSON config file (required) |
| `--source_rep` | Source shape representation: `mesh`, `sdf`, or `pt` |
| `--target_rep` | Target shape representation: `mesh`, `sdf`, or `pt` |
| `--mesh_baseline` | Load the mesh to compute baselines requiring mesh connectivity |
| `--matching_methods` | Method group to evaluate (see table above) |
| `--source_shape` | Evaluate only from this source shape |
| `--target_shape` | Evaluate only to this target shape |
| `--same` | Match each shape to itself (identity, for sanity checking) |
| `--backward_steps` | ODE integration steps for the inverse flow (default: 64) |
| `--forward_steps` | ODE integration steps for the forward flow (default: 64) |
| `--features_normalization` | Apply an additional normalization at evaluation time |
| `--mlp_hidden_size` | MLP hidden size (must match the trained model) |
| `--mlp_depth` | MLP depth (must match the trained model) |
| `--mlp_num_frequencies` | Fourier frequencies (must match the trained model) |
| `--network` | Network backbone (must match the trained model) |
| `--edm_preconditioning` | Apply EDM-style sigma preconditioning (must match the training setting) |
| `--plot_png` | Save PNG visualizations of correspondences |
| `--plot_html` | Save interactive HTML visualizations |
| `--matching_run_name` | Name suffix for the output directory |

### Outputs

Results are saved in `out/matching/<dataset>/matching-<source_rep>-<target_rep>-<run_name>/`:
- `matching_results.csv` — per-pair, per-method metrics (appended incrementally)
- `matching_results_completed.csv` — full results after all pairs complete
- `matching_results_average.csv` — metrics averaged over all pairs, grouped by method
- `timing_results.txt` — per-pair wall-clock time
- `p2p/` — saved point-to-point correspondence arrays (`.npy`)
- Visualization plots (if `--plot_png` or `--plot_html` are set)

---

## Configuration and Dataset Paths

All dataset paths, output directories, and landmark indices are configured
through a single JSON file. A fully annotated template is provided in
[`sample-config.json`](sample-config.json), note that every `/path/to/...`
entry should be updated to match your local setup before running
any script.

The config has two main sections:

- **Top-level training fields**: used by `main.py` for single-shape training
  (method, network architecture, features, normalization, paths for one shape).
- **`matching_config`**: one sub-object per dataset, each containing the
  dataset mesh folder, geodesic distance cache folder, trained flow output
  folder, correspondence file folder, and landmark indices. These are read by
  `matching.py` and the `scripts/run_*.py` dataset scripts.

Precomputed geodesic distance matrices are cached in the folder pointed to by
each dataset's `dists_path` key. The first run for a given shape computes and
saves the full NxN distance matrix; all subsequent runs load it from disk.

The `landmarks` field specifies vertex indices on the reference mesh used to
compute the geodesic descriptor for each point. For SHREC20 and TOSCA, where
landmarks vary per shape, set `"landmarks": [-1, -1, -1, -1, -1, -1]` and
provide `common_landmarks_path` (SHREC20) or `corr_path` (TOSCA) so the script
can resolve per-shape landmarks at runtime.
