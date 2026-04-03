from pathlib import Path

import pandas as pd

from matching.data_structures import DataPath
from matching.element_processing import process_element
from matching.evaluation import log_results, run_matching_methods
from matching.methods import METHOD_GROUPS, get_matching_methods
from matching.visualization import plot_results


def process_pair(
    source: str,
    target: str,
    source_rep: str,
    target_rep: str,
    device: str,
    mesh_baseline: bool,
    plot_html: bool,
    plot_png: bool,
    selected_methods: str,
    features_normalization: str,
    data_path: "DataPath",
    output_dir: str,
    backward_steps: int,
    forward_steps: int,
    embedding_dim: int,
    mlp_hidden_size: int = 256,
    mlp_depth: int = 4,
    mlp_num_frequencies: int = -1,
    network_selection: str = "MLP",
    edm_preconditioning: bool = False,
) -> pd.DataFrame:
    """
    Main pipeline to process a source-target pair for shape matching.
    Handles loading, feature extraction, matching, evaluation, and visualization.

    Args:
        source, target (str): Source and target element identifiers.
        source_rep, target_rep (str): Representation types ('mesh', 'sdf', or 'pt').
        device (str): Device to use for computation.
        mesh_baseline (bool): Whether to use the mesh as the base geometry.
        plot_html, plot_png (bool): Whether to export visualization plots.
        selected_methods (str): Matching methods to evaluate.
        features_normalization (str): Feature normalization scheme.
        data_path (DataPath): Container for dataset paths.
        output_dir (str): Directory to store results.
        embedding_dim (int): Number of landmark features / flow channels.

    Returns:
        pd.DataFrame: Results table containing evaluation metrics for selected methods.
    """

    load_flow = any(m.startswith("FUSE") for m in METHOD_GROUPS[selected_methods])

    source_element = process_element(
        element=source,
        representation=source_rep,
        device=device,
        mesh_baseline=mesh_baseline,
        features_normalization=features_normalization,
        data_path=data_path,
        embedding_dim=embedding_dim,
        mlp_hidden_size=mlp_hidden_size,
        mlp_depth=mlp_depth,
        mlp_num_frequencies=mlp_num_frequencies,
        network_selection=network_selection,
        edm_preconditioning=edm_preconditioning,
        load_flow=load_flow,
    )

    target_element = process_element(
        element=target,
        representation=target_rep,
        device=device,
        mesh_baseline=mesh_baseline,
        features_normalization=features_normalization,
        data_path=data_path,
        embedding_dim=embedding_dim,
        mlp_hidden_size=mlp_hidden_size,
        mlp_depth=mlp_depth,
        mlp_num_frequencies=mlp_num_frequencies,
        network_selection=network_selection,
        edm_preconditioning=edm_preconditioning,
        load_flow=load_flow,
    )

    matching_methods = get_matching_methods(
        source_features=source_element.vertex_features,
        target_features=target_element.vertex_features,
        source_path=Path(data_path.dataset_path, source + data_path.dataset_extension),
        target_path=Path(data_path.dataset_path, target + data_path.dataset_extension),
        source_model=source_element.model,
        target_model=target_element.model,
        source_landmarks=source_element.landmarks,
        target_landmarks=target_element.landmarks,
        device=device,
        matching_methods=selected_methods,
        backward_steps=backward_steps,
        forward_steps=forward_steps,
        source_sdf_projected_vertex_points=source_element.vertex_points,
        target_sdf_projected_vertex_points=target_element.vertex_points,
    )

    results = run_matching_methods(
        matching_methods=matching_methods,
        target_points=target_element.vertex_points,
        source_element=source_element,
        target_element=target_element,
        target_element_dists=target_element.dists,
        output_dir=output_dir,
        gts_path=data_path.gts_path,
        data_path=data_path,
    )

    log_results(source, target, results)

    plot_results(
        source_points=source_element.vertex_points,
        target_points=target_element.vertex_points,
        source=source,
        target=target,
        results=results,
        output_dir=str(data_path.output_dir),
        plot_html=plot_html,
        plot_png=plot_png,
        max_points=100_000,
    )

    df = pd.DataFrame(
        [
            {
                "source": source,
                "target": target,
                "method": name,
                "euclidean_error": res.euclidean_error,
                "geodesic_error": res.geodesic_error,
                "dirichlet": res.dirichlet_energy,
                "coverage": res.coverage,
                "elapsed": res.elapsed,
            }
            for name, res in results.items()
        ]
    )

    return df
