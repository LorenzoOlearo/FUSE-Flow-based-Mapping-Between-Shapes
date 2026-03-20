"""Flow composition-based point-to-point matching methods."""

import time

import numpy as np
import torch
from tqdm import tqdm

from matching.p2p.knn import compute_p2p_knn_zoomout, compute_p2p_knn_neural_zoomout


def compute_p2p_fuse(
    source_input,
    target_input,
    source_model,
    target_model,
    backward_steps: int,
    forward_steps: int,
):
    tqdm.write("> computing p2p with flows composition")
    start_time = time.time()

    with torch.no_grad():
        emb1_pullback = source_model.inverse(
            samples=source_input, num_steps=backward_steps
        )
        sample = target_model.sample(noise=emb1_pullback, num_steps=forward_steps)

    dists = torch.cdist(sample, target_input)  # [N, M]
    p2p = torch.argmin(dists, dim=1).cpu().numpy()

    elapsed_time = time.time() - start_time
    tqdm.write(f">>>> compute_p2p_fuse elapsed {elapsed_time:.4f}s")
    return p2p, elapsed_time


def compute_p2p_fuse_hungarian(
    source_input,
    target_input,
    source_model,
    target_model,
    backward_steps: int,
    forward_steps: int,
):
    from scipy.optimize import linear_sum_assignment

    tqdm.write("> computing p2p with flows composition + hungarian")
    start_time = time.time()

    with torch.no_grad():
        emb1_pullback = source_model.inverse(
            samples=source_input, num_steps=backward_steps
        )
        sample = target_model.sample(noise=emb1_pullback, num_steps=forward_steps)

    dists = torch.cdist(sample, target_input)
    row_ind, col_ind = linear_sum_assignment(dists.cpu().numpy())
    p2p = col_ind[np.argsort(row_ind)]

    elapsed_time = time.time() - start_time
    tqdm.write(f">>>> compute_p2p_fuse_hungarian elapsed {elapsed_time:.4f}s")
    return p2p, elapsed_time


def compute_p2p_fuse_lapjv(
    source_input,
    target_input,
    source_model,
    target_model,
    backward_steps: int,
    forward_steps: int,
):
    from lapjv import lapjv

    tqdm.write("> computing p2p with flows composition + lapjv")
    start_time = time.time()

    with torch.no_grad():
        emb1_pullback = source_model.inverse(
            samples=source_input, num_steps=backward_steps
        )
        sample = target_model.sample(noise=emb1_pullback, num_steps=forward_steps)

    dists = torch.cdist(sample, target_input)
    row_ind, col_ind, _ = lapjv(dists.cpu().numpy())
    p2p = col_ind[np.argsort(row_ind)]

    elapsed_time = time.time() - start_time
    tqdm.write(f">>>> compute_p2p_fuse_lapjv elapsed {elapsed_time:.4f}s")
    return p2p, elapsed_time


def compute_p2p_fuse_anchor(
    source_input,
    target_input,
    source_model,
    target_model,
    backward_steps: int,
    forward_steps: int,
):
    from sklearn.neighbors import NearestNeighbors

    tqdm.write("> computing p2p with inverted flows in gauss")
    start_time = time.time()

    with torch.no_grad():
        emb1_pullback = source_model.inverse(
            samples=source_input, num_steps=backward_steps
        )
        emb2_pullback = target_model.inverse(
            samples=target_input, num_steps=forward_steps
        )

    nbrs = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(
        emb2_pullback.cpu().numpy()
    )
    _, p2p = nbrs.kneighbors(emb1_pullback.cpu().numpy())
    p2p = p2p[:, 0]

    elapsed_time = time.time() - start_time
    tqdm.write(f">>>> compute_p2p_fuse_anchor elapsed {elapsed_time:.4f}s")
    return p2p, elapsed_time


def compute_p2p_fuse_anchor_uniformed(
    source_input, target_input, source_model, target_model
):
    from sklearn.neighbors import NearestNeighbors
    from torch.distributions import Normal

    tqdm.write("> computing p2p with inverted flows in gauss uniformed")
    start_time = time.time()

    with torch.no_grad():
        emb1_pullback = source_model.inverse(samples=source_input, num_steps=64)
        emb2_pullback = target_model.inverse(samples=target_input, num_steps=64)

    normal = Normal(0, 1)
    emb1_pullback_uniform = normal.cdf(emb1_pullback)
    emb2_pullback_uniform = normal.cdf(emb2_pullback)

    nbrs = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(
        emb2_pullback_uniform.cpu().numpy()
    )
    _, p2p = nbrs.kneighbors(emb1_pullback_uniform.cpu().numpy())
    p2p = p2p[:, 0]

    elapsed_time = time.time() - start_time
    tqdm.write(f">>>> compute_p2p_fuse_anchor_uniformed elapsed {elapsed_time:.4f}s")
    return p2p, elapsed_time


def compute_p2p_fuse_zoomout(
    source_path,
    target_path,
    source_input,
    target_input,
    source_model,
    target_model,
    backward_steps: int,
    forward_steps: int,
    device: str,
):
    tqdm.write("> computing p2p with flows composition + zoomout")
    start_time = time.time()

    with torch.no_grad():
        emb1_pullback = source_model.inverse(
            samples=source_input, num_steps=backward_steps
        )
        sample = target_model.sample(noise=emb1_pullback, num_steps=forward_steps)

    p2p, _ = compute_p2p_knn_zoomout(
        source_path, target_path, sample, target_input, device
    )

    elapsed_time = time.time() - start_time
    tqdm.write(f">>>> compute_p2p_fuse_zoomout elapsed {elapsed_time:.4f}s")
    return p2p, elapsed_time


def compute_p2p_fuse_neural_zoomout(
    source_path,
    target_path,
    source_input,
    target_input,
    source_model,
    target_model,
    backward_steps: int,
    forward_steps: int,
    device,
):
    tqdm.write("> computing p2p with flows composition + neural zoomout")
    start_time = time.time()

    with torch.no_grad():
        emb1_pullback = source_model.inverse(
            samples=source_input, num_steps=backward_steps
        )
        sample = target_model.sample(noise=emb1_pullback, num_steps=forward_steps)

    p2p, _ = compute_p2p_knn_neural_zoomout(
        source_path, target_path, sample, target_input, device
    )

    elapsed_time = time.time() - start_time
    tqdm.write(f">>>> compute_p2p_fuse_neural_zoomout elapsed {elapsed_time:.4f}s")
    return p2p, elapsed_time
