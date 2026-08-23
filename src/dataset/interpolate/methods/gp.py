import numpy as np
import polars as pl

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

from ..core import nan_runs

def fill_gaussian_process(series: pl.Series, windows: int = 120, seed: int = 7, optimize_hyperparams: bool = True, length_scale: float = 30.0, noise_level: float = 0.1, only_mask: np.ndarray = None):
    """
    Rellena huecos interiores de una serie con un Proceso Gaussiano Local.
    Para cada rango de valores faltantes que no toca los extremos de la serie, ajusta un GaussianProcessRegressor sobre una ventana de contexto válido a ambos lados del hueco y predice los valores faltantes junto con su desviación estándar. Los datos se centran y escalan antes de ajustar y se des-escalan al predecir, para estabilizar el ajuste.
    Si only_mask (booleano del largo de la serie) se entrega, solo se ajustan los huecos que caen dentro de esa máscara; el resto queda como NaN. Evita ajustar un GP por cada hueco de la columna cuando solo se necesitan algunos (huecos inyectados del benchmark, o huecos asignados a gp en modo best).
    """
    name = series.name
    values = series.to_numpy().astype("float32")
    n = values.size
    valid = np.isfinite(values)

    fill = values.copy()
    sigma = np.full(n, np.nan)
    x_all = np.arange(n, dtype = "float32")

    kernel = ConstantKernel(1.0) * RBF(length_scale = length_scale) + WhiteKernel(noise_level = noise_level)
    optimizer = "fmin_l_bfgs_b" if optimize_hyperparams else None

    for start, length in nan_runs(~valid):
        if start == 0 or start + length == n:
            continue
        if only_mask is not None and not only_mask[start:start + length].any():
            continue

        lo = max(0, start - windows)
        hi = min(n, start + length + windows)
        ctx_mask = valid[lo:hi]

        x_ctx = x_all[lo:hi][ctx_mask]
        y_ctx = values[lo:hi][ctx_mask]

        if x_ctx.size < 3:
            continue

        x0 = x_ctx.mean()
        y0 = y_ctx.mean()
        y_std = y_ctx.std() if y_ctx.std() > 0 else 1.0

        gp = GaussianProcessRegressor(
            kernel = kernel,
            normalize_y = False,
            random_state = seed,
            n_restarts_optimizer = 0,
            optimizer = optimizer,
        )

        gp.fit((x_ctx - x0).reshape(-1, 1), (y_ctx - y0) / y_std)

        x_gap = x_all[start:start + length]
        mu, sd = gp.predict((x_gap - x0).reshape(-1, 1), return_std = True)

        fill[start:start + length] = mu * y_std + y0
        sigma[start: start + length] = sd * y_std

    series_out = pl.Series(name, fill)
    sigma_out = pl.Series(f"{name}_sigma", sigma)

    return series_out, sigma_out