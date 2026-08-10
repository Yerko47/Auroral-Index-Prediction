import numpy as np
import pandas as pd

from .core import nan_runs, bin_labels, assign_bin, auto_gap_bins, gap_length_distribution
from .methods import fill_pchip, fill_gaussian_process

def inject_syntetic_gaps(series, gap_lengths, n_gaps, seed = 7, margen = 5):
    """
    """
    rng = np.random.default_rng(seed)

    n = series.size
    gappeada = series.copy()
    inyected = np.zeros(n, dtype = bool)

    if gap_lengths.size == 0: return gappeada, inyected

    occupied = np.zeros(n, dtype = bool)
    