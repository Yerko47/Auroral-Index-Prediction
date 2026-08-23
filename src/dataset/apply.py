from datetime import datetime

import numpy as np
import polars as pl

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

from .interpolate.core import missing_mask, nan_runs, bin_labels, assign_bin, gap_length_distribution
from .interpolate.methods import fill_pchip, fill_gaussian_process, fill_iterative
from .interpolate.benchmark import benchmark_methods, bins_from_lengths


def _interior_gaps(mask):
    """
    Devuelve los gaps (start, length) que no tocan los bordes de la serie.
    """
    n = mask.size
    out = []
    
    for start, length in nan_runs(mask):
        if start == 0 or start + length == n:
            continue
        out.append((start, length))

    return out


def _winners_by_column_bin(res):
    """
    Del resultado del benchmark, elige el método de menor RMSE por (columna, bin).
    """
    win = {}

    for r in res.filter(pl.col("bin") != "all").iter_rows(named = True):
        key = (r["column"], r["bin"])
        if key not in win or r["rmse"] < win[key][1]:
            win[key] = (r["method"], r["rmse"])

    for r in res.filter(pl.col("bin") == "all").iter_rows(named = True):
        win.setdefault((r["column"], "all"), (r["method"], r["rmse"]))

    return {k: v[0] for k, v in win.items()}

def _choose_method(method, col, length, icfg, winners, bins):
    """Resuelve qué método aplicar a un gap concreto según la config."""
    if method in ("pchip", "gp", "iterative"):
        return method
    if method == "auto":
        return "pchip" if length <= icfg["umbral_corto"] else "iterative"
    if method == "best":
        lab = bin_labels(bins)[assign_bin(length, bins)]
        return winners.get((col, lab)) or winners.get((col, "all")) or "pchip"
    raise ValueError(f"metodo de interpolacion desconocido: {method}")


def apply_interpolation(df, cfg, benchmark_result=None, seed=7):
    """
    Aplica la interpolación a las columnas flotantes según cfg["interpolation"].
    Rellena solo los gaps interiores. Cada gap se rellena con el método fijo (pchip/gp/iterative), por umbral (auto) o el ganador del benchmark por variable y bin (best). Devuelve el DataFrame interpolado (sin escalar) y una tabla de procedencia con una fila por zona: column, start, length, t_start, t_end, method.
    """
    icfg = cfg["interpolation"]
    method = icfg["method"]
    target = [c for c, dt in df.schema.items() if dt.is_float()]

    winners = None
    bins = None
    if method == "best":
        gl = gap_length_distribution(df, columns=target)
        gl = gl if gl.size > 0 else np.array([1, 2, 3, 5, 10, 20])
        bins = bins_from_lengths(gl, icfg.get("n_bins", 4))
        if benchmark_result is None:
            benchmark_result = benchmark_methods(
                df, target_columns=target, feature_columns=target,
                n_gaps=icfg.get("n_gaps_por_repeticion", 40),
                gap_lengths=gl, n_bins=icfg.get("n_bins", 4), seed=seed,
            )
        winners = _winners_by_column_bin(benchmark_result)

    iter_df = None
    if method in ("iterative", "auto", "best"):
        iter_df = fill_iterative(df, columns=target, seed=seed)

    t = df["Epoch"].to_numpy() if "Epoch" in df.columns else np.arange(len(df))
    result = df.clone()
    applied = []

    for col in target:
        mask = missing_mask(df[col])
        gaps = _interior_gaps(mask)
        if not gaps:
            continue

        out = df[col].to_numpy().astype("float32")
        cache = {}

        def get_fill(m):
            if m == "iterative":
                return iter_df[col].to_numpy()
            if m not in cache:
                cache[m] = (fill_pchip(df[col]).to_numpy() if m == "pchip"
                            else fill_gaussian_process(df[col], seed=seed)[0].to_numpy())
            return cache[m]

        for start, length in gaps:
            m = _choose_method(method, col, length, icfg, winners, bins)
            out[start:start + length] = get_fill(m)[start:start + length]
            applied.append({
                "column": col, "start": int(start), "length": int(length),
                "t_start": t[start], "t_end": t[start + length - 1], "method": m,
            })
        result = result.with_columns(pl.Series(col, out))

    schema = {"column": pl.Utf8, "start": pl.Int64, "length": pl.Int64,
              "t_start": pl.Datetime, "t_end": pl.Datetime, "method": pl.Utf8}
    provenance = pl.DataFrame(applied) if applied else pl.DataFrame(schema=schema)
    return result, provenance


def apply_scaling(df, cfg, columns=None):
    """
    Escala las columnas flotantes con un scaler de scikit-learn.
    El método (standard, robust, minmax) se toma de cfg["scaling"]["method"]. Los scalers de sklearn ignoran NaN en fit y los preservan en transform, por lo que los gaps de borde no rellenados quedan como NaN tras escalar.
    """
    scfg = cfg.get("scaling", {})
    method = scfg.get("method", "standard")
    scaler_type = {"standard": StandardScaler, "robust": RobustScaler, "minmax": MinMaxScaler}
    if method not in scaler_type:
        raise ValueError(f"scaler desconocido: {method}")

    columns = columns or [c for c, dt in df.schema.items() if dt.is_float()]
    X = df.select(columns).to_numpy().astype("float64")

    scaler = scaler_type[method]()
    Xs = scaler.fit_transform(X)   # NaN se ignoran en fit y se preservan en transform

    scaled = df.with_columns(
        [pl.Series(c, Xs[:, j].astype("float32")) for j, c in enumerate(columns)]
    )
    return scaled, scaler


def save_interpolation(df_interp: pl.DataFrame, provenance: pl.DataFrame, cfg: dict, paths: dict, debug_mode: bool, logging_info = None) -> None:
    """
    Guarda el DataFrame interpolado (sin escalar) y su tabla de procedencia en data/processed.
    El interpolado se escribe en formato feather; la procedencia (una fila por zona rellenada: column, start, length, t_start, t_end, method) se escribe en un archivo aparte. Los nombres siguen el patrón del raw, usando el rango de fechas de cfg["dataset"]["time_range"].
    """
    start_time = datetime.fromisoformat(cfg["dataset"]["time_range"]["start"])
    end_time = datetime.fromisoformat(cfg["dataset"]["time_range"]["end"])

    interp_file = paths["processed_file"] / f"interp_{start_time.year}_to_{end_time.year}.feather"
    provenance_file = paths["processed_file"] / f"provenance_{start_time.year}_to_{end_time.year}.feather"

    df_interp.write_ipc(interp_file)
    provenance.write_ipc(provenance_file)

    if logging_info is not None:
        logging_info(debug_mode,
            f"Interpolated data saved in {interp_file}\n"
            f"Provenance saved in {provenance_file}\n"
            + "=" * 70
        )