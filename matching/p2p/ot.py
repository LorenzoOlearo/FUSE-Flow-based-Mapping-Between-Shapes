"""Optimal transport point-to-point matching."""

import time

import numpy as np
import ot
from tqdm import tqdm


def compute_p2p_ot(source_features, target_features):
    tqdm.write("> computing p2p with ot (sinkhorn)")
    start_time = time.time()

    M = np.exp(-ot.dist(source_features.cpu().numpy(), target_features.cpu().numpy()))

    n, m = M.shape
    a = np.ones(n) / n
    b = np.ones(m) / m

    Gs = ot.sinkhorn(a, b, M, reg=1)

    indices = np.argsort(Gs, axis=1)[:, :1]
    p2p = indices.T[0, :]

    elapsed_time = time.time() - start_time
    tqdm.write(f">>>> compute_p2p_ot elapsed {elapsed_time:.4f}s")
    return p2p, elapsed_time
