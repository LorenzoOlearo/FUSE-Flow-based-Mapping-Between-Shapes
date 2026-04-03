from matching.p2p import (
    compute_p2p_fmaps,
    compute_p2p_fmaps_neural_zoomout,
    compute_p2p_fmaps_zoomout,
    compute_p2p_fuse,
    compute_p2p_fuse_anchor,
    compute_p2p_fuse_hungarian,
    compute_p2p_fuse_lapjv,
    compute_p2p_fuse_neural_zoomout,
    compute_p2p_fuse_zoomout,
    compute_p2p_hungarian,
    compute_p2p_knn,
    compute_p2p_lapjv,
    compute_p2p_ndp_landmarks,
    compute_p2p_ndp_sdf,
    compute_p2p_ndp_wks,
    compute_p2p_ot,
)

METHOD_GROUPS = {
    "fast": ["KNN", "FUSE"],
    "sdf": ["KNN", "OT", "FUSE", "NDP-SDF", "FUSE-ANCHOR"],
    "all": [
        "KNN",
        "OT",
        "FMaps",
        "FMaps-zoomout",
        "FMaps-neural-zoomout",
        "NDP-landmarks",
        "FUSE",
        "FUSE-ANCHOR",
        "FUSE-zoomout",
        "FUSE-neural-zoomout",
    ],
    "baselines": [
        "KNN",
        "OT",
        "FUSE",
        "FMaps",
        "FMaps-zoomout",
        "FMaps-neural-zoomout",
        "FUSE-ANCHOR",
        "NDP-landmarks",
    ],
    "baselines-no-zoomout": [
        "KNN",
        "OT",
        "FUSE",
        "FMaps",
        "FUSE-ANCHOR",
        "NDP-landmarks",
    ],
    "zoomout": ["FMaps", "FMaps-zoomout", "FMaps-neural-zoomout"],
    "la": ["KNN", "FUSE", "hungarian", "lapjv", "FUSE-hungarian", "FUSE-lapjv"],
    "baselines-no-FUSE": [
        "KNN",
        "OT",
        "FMaps",
        "FMaps-zoomout",
        "FMaps-neural-zoomout",
        "NDP-landmarks",
    ],
}


def get_matching_methods(
    source_features,
    target_features,
    source_path,
    target_path,
    source_model,
    target_model,
    source_landmarks,
    target_landmarks,
    device: str,
    matching_methods: str,
    backward_steps: int,
    forward_steps: int,
    source_sdf_projected_vertex_points=None,
    target_sdf_projected_vertex_points=None,
):
    """Return mapping of strategy names to their compute functions."""

    source_features = source_features.to(device)
    target_features = target_features.to(device)
    if source_model is not None:
        source_model = source_model.to(device)
    if target_model is not None:
        target_model = target_model.to(device)

    all_methods = {
        "KNN": lambda: compute_p2p_knn(source_features, target_features),
        "OT": lambda: compute_p2p_ot(source_features, target_features),
        "FUSE": lambda: compute_p2p_fuse(
            source_features,
            target_features,
            source_model,
            target_model,
            backward_steps=backward_steps,
            forward_steps=forward_steps,
        ),
        "FMaps": lambda: compute_p2p_fmaps(
            source_path, target_path, source_features, target_features
        ),
        "FMaps-zoomout": lambda: compute_p2p_fmaps_zoomout(
            source_path, target_path, source_features, target_features, device
        ),
        "FMaps-neural-zoomout": lambda: compute_p2p_fmaps_neural_zoomout(
            source_path, target_path, source_features, target_features, device
        ),
        "NDP-landmarks": lambda: compute_p2p_ndp_landmarks(
            source_path,
            target_path,
            source_landmarks=source_landmarks,
            target_landmarks=target_landmarks,
        ),
        "NDP-wks": lambda: compute_p2p_ndp_wks(
            source_path,
            target_path,
            source_landmarks=source_landmarks,
            target_landmarks=target_landmarks,
        ),
        "NDP-SDF": lambda: compute_p2p_ndp_sdf(
            source_vertex=source_sdf_projected_vertex_points,
            target_vertex=target_sdf_projected_vertex_points,
            source_landmarks=source_landmarks,
            target_landmarks=target_landmarks,
        ),
        "FUSE-ANCHOR": lambda: compute_p2p_fuse_anchor(
            source_features,
            target_features,
            source_model,
            target_model,
            backward_steps=backward_steps,
            forward_steps=forward_steps,
        ),
        "FUSE-zoomout": lambda: compute_p2p_fuse_zoomout(
            source_path,
            target_path,
            source_features,
            target_features,
            source_model,
            target_model,
            backward_steps=backward_steps,
            forward_steps=forward_steps,
            device=device,
        ),
        "FUSE-neural-zoomout": lambda: compute_p2p_fuse_neural_zoomout(
            source_path,
            target_path,
            source_features,
            target_features,
            source_model,
            target_model,
            backward_steps=backward_steps,
            forward_steps=forward_steps,
            device=device,
        ),
        "FUSE-hungarian": lambda: compute_p2p_fuse_hungarian(
            source_features,
            target_features,
            source_model,
            target_model,
            backward_steps=backward_steps,
            forward_steps=forward_steps,
        ),
        "FUSE-lapjv": lambda: compute_p2p_fuse_lapjv(
            source_features,
            target_features,
            source_model,
            target_model,
            backward_steps=backward_steps,
            forward_steps=forward_steps,
        ),
        "hungarian": lambda: compute_p2p_hungarian(
            source_features,
            target_features,
        ),
        "lapjv": lambda: compute_p2p_lapjv(
            source_features,
            target_features,
        ),
    }

    if matching_methods not in METHOD_GROUPS:
        raise ValueError(f"Unknown matching methods option: {matching_methods}")

    selected_keys = METHOD_GROUPS[matching_methods]
    return {key: all_methods[key] for key in selected_keys}
