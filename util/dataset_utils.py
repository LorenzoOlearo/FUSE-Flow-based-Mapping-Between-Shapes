import argparse
import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import trimesh
from scipy.io import loadmat


def get_common_landmarks_between_two_models(model1_mat_path, model2_mat_path):
    """
    Given two SHREC20 ground truth .mat files (each containing 'points' and 'verts'),
    find the common landmark IDs between them and return the aligned vertex indices.

    Parameters
    ----------
    model1_mat_path : str
        Path to the .mat file for model 1.
    model2_mat_path : str
        Path to the .mat file for model 2.

    Returns
    -------
    common_ids : np.ndarray
        Sorted list of common landmark IDs (0-based).
    model1_common_idxs : np.ndarray
        Vertex indices in model 1 corresponding to common_ids.
    model2_common_idxs : np.ndarray
        Vertex indices in model 2 corresponding to common_ids.
    """

    # Load the two .mat files
    gt1 = loadmat(model1_mat_path)
    gt2 = loadmat(model2_mat_path)

    # Extract landmark IDs and corresponding vertex indices (convert to 0-based)
    ids1 = np.asarray(gt1["points"]).astype(int).flatten() - 1
    idxs1 = np.asarray(gt1["verts"]).astype(int).flatten() - 1

    ids2 = np.asarray(gt2["points"]).astype(int).flatten() - 1
    idxs2 = np.asarray(gt2["verts"]).astype(int).flatten() - 1

    # Find intersection (common landmark IDs)
    common_ids = sorted(list(set(ids1).intersection(set(ids2))))
    common_ids = np.array(common_ids, dtype=int)

    if len(common_ids) == 0:
        print("No common landmark IDs found between the two models.")
        exit(1)

    # Build id -> vertex index mapping for each model
    id_to_idx_1 = dict(zip(ids1.tolist(), idxs1.tolist()))
    id_to_idx_2 = dict(zip(ids2.tolist(), idxs2.tolist()))

    # Extract vertex indices aligned by the same common_id order
    model1_common_idxs = np.array([id_to_idx_1[cid] for cid in common_ids], dtype=int)
    model2_common_idxs = np.array([id_to_idx_2[cid] for cid in common_ids], dtype=int)

    return common_ids, model1_common_idxs, model2_common_idxs


def get_shrec20_landmarks(models_path, gts_path, plot):
    """
    Finds the largest common set of landmark IDs across all SHREC20 models
    and returns per-model vertex indices aligned in the same ID order.
    """
    landmarks_id_all = {}
    landmarks_idx_all = {}
    gts = {}

    for model in os.listdir(models_path):
        if model.endswith(".obj"):
            print(f"Processing model: {model}")
            gt_file = os.path.join(gts_path, model.replace(".obj", ".mat"))
            gt = loadmat(gt_file)
            gts[model] = gt

            # ensure shapes are (N,) and convert to 0-based
            landmarks_id = np.asarray(gt["points"]).astype(int).flatten() - 1
            landmarks_idx = np.asarray(gt["verts"]).astype(int).flatten() - 1

            landmarks_id_all[model] = landmarks_id
            landmarks_idx_all[model] = landmarks_idx

    # compute common IDs (as sorted list -> deterministic order)
    model_names = list(landmarks_id_all.keys())
    if len(model_names) == 0:
        return {}

    common_ids = set(landmarks_id_all[model_names[0]])
    for ids in landmarks_id_all.values():
        common_ids.intersection_update(set(ids))
    common_ids = sorted(list(common_ids))
    print(f"Found {len(common_ids)} common landmark IDs across all models.")

    # produce per-model aligned vertex indices (ordered according to common_ids)
    landmarks_idx_all_common = {}
    for model in model_names:
        ids = landmarks_id_all[model]
        idxs = landmarks_idx_all[model]

        # build mapping id -> vertex index for this model
        id_to_idx = dict(zip(ids.tolist(), idxs.tolist()))

        # ensure every common_id is present in mapping (sanity)
        missing = [cid for cid in common_ids if cid not in id_to_idx]
        if missing:
            # should not happen because we built common_ids via intersection, but check anyway
            raise RuntimeError(
                f"Model {model} missing common IDs: {missing[:10]} (showing up to 10)"
            )

        # create aligned array of vertex indices in the same order as common_ids
        aligned_idxs = np.array([id_to_idx[cid] for cid in common_ids], dtype=int)
        landmarks_idx_all_common[model] = aligned_idxs

        # (optional) quick plot of aligned landmarks for visual check
        if plot == True:
            mesh = trimesh.load(os.path.join(models_path, model))
            landmark_points = mesh.vertices[aligned_idxs]
            fig = go.Figure(
                data=[
                    go.Mesh3d(
                        x=mesh.vertices[:, 0],
                        y=mesh.vertices[:, 1],
                        z=mesh.vertices[:, 2],
                        i=mesh.faces[:, 0],
                        j=mesh.faces[:, 1],
                        k=mesh.faces[:, 2],
                        color="lightgrey",
                        opacity=0.5,
                    ),
                    go.Scatter3d(
                        x=landmark_points[:, 0],
                        y=landmark_points[:, 1],
                        z=landmark_points[:, 2],
                        mode="markers+text",
                        marker=dict(size=4, color="red"),
                        text=[
                            str(common_ids[i]) for i in range(len(common_ids))
                        ],  # index in the common_ids list
                        textposition="top center",
                    ),
                ]
            )
            fig.update_layout(
                title=f"Aligned Common Landmarks on {model}",
                scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
            )

            output_dir = "./plots/shrec20_common_landmarks/"
            os.makedirs(output_dir, exist_ok=True)
            fig.write_html(os.path.join(output_dir, f"{model}_common_landmarks.html"))

    # Build a DataFrame: rows = models, columns = common landmark IDs, values = vertex indices
    common_landmarks_df = (
        pd.DataFrame(
            [landmarks_idx_all_common[m] for m in model_names],
            index=model_names,
            columns=[int(cid) for cid in common_ids],
        )
        .reset_index()
        .rename(columns={"index": "Model"})
    )

    # Save as CSV
    output_csv_path = "./plots/shrec20_common_landmarks/common_landmarks_idx.csv"
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    common_landmarks_df.to_csv(output_csv_path, index=False)
    print(f"Saved common landmark indices to {output_csv_path}")

    return common_landmarks_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Matching experiments for SDFs and meshes"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to the config file",
        required=True,
        default="config.json",
    )
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = json.load(f)

    models_path = config["matching_config"]["SHREC20"]["dataset_path"]
    gts_path = config["matching_config"]["SHREC20"]["gts_path"]

    common_landmarks_df = get_shrec20_landmarks(
        models_path=models_path, gts_path=gts_path, plot=True
    )

    landmark_head_id = 9
    landmark_right_front_foot_id = 19
    landmark_right_back_foot_id = 48
    landmark_back_id = 41
    landmark_left_front_leg = 16
    landmark_back_top = 32

    # Get these columns from the DataFrame
    selected_landmarks = common_landmarks_df[
        [
            "Model",
            landmark_head_id,
            landmark_right_front_foot_id,
            landmark_right_back_foot_id,
            landmark_back_id,
            landmark_left_front_leg,
            landmark_back_top,
        ]
    ]
    print(selected_landmarks)
    output_selected_csv_path = config["matching_config"]["SHREC20"][
        "common_landmarks_path"
    ]
    selected_landmarks.to_csv(output_selected_csv_path, index=False)
    print(f"Saved selected common landmark indices to {output_selected_csv_path}")
