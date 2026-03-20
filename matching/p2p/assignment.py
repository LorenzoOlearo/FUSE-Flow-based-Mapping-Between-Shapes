"""Linear assignment point-to-point matching methods (Hungarian, LAPJV)."""

import time

import numpy as np
import torch
from lapjv import lapjv
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm


def compute_p2p_hungarian(source_input, target_input):
    tqdm.write("> computing p2p with hungarian")
    start_time = time.time()

    dists = torch.cdist(source_input, target_input)
    row_ind, col_ind = linear_sum_assignment(dists.cpu().numpy())
    p2p = col_ind[np.argsort(row_ind)]

    elapsed_time = time.time() - start_time
    tqdm.write(f">>>> compute_p2p_hungarian elapsed {elapsed_time:.4f}s")
    return p2p, elapsed_time


def compute_p2p_lapjv(source_input, target_input):
    tqdm.write("> computing p2p with lapjv")
    start_time = time.time()

    dists = torch.cdist(source_input, target_input)
    row_ind, col_ind, _ = lapjv(dists.cpu().numpy())
    p2p = col_ind[np.argsort(row_ind)]

    elapsed_time = time.time() - start_time
    tqdm.write(f">>>> compute_p2p_lapjv elapsed {elapsed_time:.4f}s")
    return p2p, elapsed_time
