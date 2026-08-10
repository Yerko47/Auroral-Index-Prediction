import numpy as np
import pandas as pd

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

from ..core import nan_runs

def fill_gaussian_process(series, windows = 120, seed = 7):
    """
    """
    values = series.to_numpy(dtype = "float32")
    n = values.size
    valid = np.isfinite(values)
    fill = values.copy()

    sigma = np.full(n, np.nan)
    x_all = np.arange(n, dtype = "float32")

    for start, length in nan_runs(~valid):
        if start == 0 or start + length == n: continue

        lo = max(0, start - windows)
        hi = min(n, start + length + windows)
        ctx_mask = valid[lo:hi]

        x_ctx = x_all[lo:hi][ctx_mask]
        y_ctx = values[lo:hi][ctx_mask]

        if x_ctx.size < 3: continue

        x0 = x_ctx.mean()
        y0 = y_ctx.mean()

        y_std = y_ctx.std() if y_ctx.std() > 0 else 1.0

        kernel = (ConstantKernel(1.0) * RBF(length_scale = 30.0) + WhiteKernel(noise_level = 0.1))
        gp = GaussianProcessRegressor(kernel = kernel, normalize_y = False, random_state = seed, n_restarts_optimizer = 0)
        gp.fit((x_ctx - x0).reshape(-1, 1), (y_ctx - y0) / y_std)

        x_gap = x_all[start:start + length]

        mu, sd = gp.predict((x_gap - x0).reshape(-1, 1), return_std = True)

        fill[start:start + length] = mu * y_std + y0
        sigma[start:start + length] = sd * y_std

    series_out = pd.Series(fill, index = series.index, name = series.name)
    sigma_out = pd.Series(sigma, index = series.index, name = f"{series.name}_sigma")

    return series_out, sigma_out