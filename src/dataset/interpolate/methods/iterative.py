import numpy as np
import polars as pl

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

def fill_iterative(df: pl.DataFrame, columns: list[str] = None, max_iter: int = 10, seed: int = 7, standardize = True):
    """
    Imputa faltantes de varias variables con IterativeImputer (MICE).
    Modela cada columna con faltantes como una regresión de las demás y repite el ajuste de forma iterativa hasta max_iter (o hasta converger). A diferencia de los métodos temporales (pchip, gp), usa la correlación entre variables en el mismo instante, no los vecinos temporales de una variable.
    """
    if columns is None:
        columns = [c for c, dt in df.schema.items() if dt.is_float()]

    X = df.select(columns).to_numpy().astype("float64")
    X[~np.isfinite(X)] = np.nan

    if standardize:
        mu = np.nanmean(X, axis = 0)
        sd = np.nanstd(X, axis = 0)
        sd = np.where(sd > 0, sd, 1.0)
        Xs = (X - mu) / sd
    else:
        Xs = X

    imputer = IterativeImputer(max_iter = max_iter, random_state = seed)
    Xi = imputer.fit_transform(Xs)

    if standardize:
        Xi = Xi * sd + mu

    return df.with_columns(
        [pl.Series(c, Xi[:,j].astype("float32")) for j, c in enumerate(columns)]
    )