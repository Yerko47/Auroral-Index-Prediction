import numpy as np
import pandas as pd

from scipy.interpolate import PchipInterpolator

def fill_pchip(series):
    """
    """
    values = series.to_numpy(dtype = "float32")
    valid = np.isfinite(values)

    if valid.sum() < 2: return series.copy()

    x = np.arange(values.size, dtype = "float32")
    interpolator = PchipInterpolator(x[valid], values[valid], extrapolate = False)

    fill = values.copy()
    missing = ~values
    fill[missing] = interpolator(x[missing])

    return pd.Series(fill, index = series.index, name = series.name)