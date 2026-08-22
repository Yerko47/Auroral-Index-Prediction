import numpy as np
import polars as pl

from scipy.interpolate import PchipInterpolator

def fill_pchip(series: pl.Series):
    """
    Rellena huevos interiores de una serie con interpolación PCHIP.
    Ajusta un interpolador cúbico de Hermite monótono por tramos sobre los puntos válidos y evalúa en la posición faltantes. Esto no extrapola, ya que los huecos que tocan el inicio o al final de la serie quedan NaN.
    """
    values = series.to_numpy().astype("float32")
    valid = np.isfinite(values)

    if valid.sum() < 2:
        return series.clone()

    x = np.arange(values.size, dtype = "float32")
    interpolator = PchipInterpolator(x[valid], values[valid], extrapolate = False)

    fill = values.copy()
    missing = ~valid

    fill[missing] = interpolator(x[missing])

    return pl.Series(series.name, fill)