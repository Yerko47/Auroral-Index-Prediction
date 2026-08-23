import numpy as np
import polars as pl

from .core import nan_runs, bin_labels, assign_bin, gap_length_distribution
from .methods import fill_pchip, fill_gaussian_process, fill_iterative

def bins_from_lengths(gap_lengths: np.array, n_bins: int = 4) -> list[float]:
    """
    Construye bordes de intervalo (escala geométrica) desde los largos inyectados.
    Los bins se derivan de los largos de hueco usados en el benchmark, no de la distribución de huecos reales del DataFrame (que en un tramo limpio puede no tener huecos). Espejo de la lógica de auto_gap_bins, aplicada a un arreglo.
    """
    lmax = int((np.max(gap_lengths)))
    if lmax <= 1:
        return [1, np.inf]

    pts = np.unique(np.round(np.geomspace(1, lmax, n_bins + 1)).astype(int))
    pts = pts[pts >= 1]
    if pts.size and pts[0] != 1:
        pts = np.insert(pts, 0 , 1)

    return [*[int(b) for b in pts], np.inf]


def inject_synthetic_gaps(valid_mask: np.ndarray, gap_lengths: np.array, n_gaps: int, rng: np.random.Generator, margin:int = 5) -> np.ndarray:
    """
    Elige posiciones válidas donde inyectar huecos sintéticos.
    Solo inyecta en tramos completamente válidos y rodeados por un margen también válido, para (a) conocer la verdad en las posiciones ocultadas y (b) garantizar contexto a ambos lados para los métodos temporales.
    """
    n = valid_mask.size
    injected = np.zeros(n, dtype = bool)
    if len(gap_lengths) == 0 or n_gaps <= 0:
        return injected

    placed = 0
    attempts = 0
    cap = n_gaps * 100

    while placed < n_gaps and attempts < cap:
        attempts += 1
        length = int(rng.choice(gap_lengths))
        if n - length - margin <= margin:
            continue
        start = int(rng.integers(margin, n - length - margin))
        window = slice(start - margin, start + length + margin)

        if valid_mask[window].all() and not injected[window].any():
            injected[start:start + length] = True
            placed += 1

    return injected


def _metrics(true, pred):
    """
    Calcula RMSE, MAE, bias y coverage sobre pares (verdad, predicción)
    """
    err = pred - true
    finite = np.isfinite(err)
    e = err[finite]
    if e.size == 0:
        return {
            "n": 0,
            "rmse": np.nan,
            "mae": np.nan,
            "bias": np.nan,
            "coverage": np.nan,
        }
    return {
        "n": int(finite.sum()),
        "rmse": float(np.sqrt(np.mean(e ** 2))),
        "mae": float(np.mean(np.abs(e))),
        "bias": float(np.mean(e)),
        "coverage": float(finite.mean()),
    }


def _scored_rows(method, col, true, pred, injected, bins, labels):
    """
    Genera filas de métricas globales y por bin de largo para un método y columna.
    """
    rows = [{
        "method": method,
        "column": col,
        "bin": "all",
        **_metrics(true[injected], pred[injected])
    }]

    by_bin = {lab: [] for lab in labels}
    for start, length in nan_runs(injected):
        by_bin[labels[assign_bin(length, bins)]].extend(range(start, start + length))

    for lab, idx in by_bin.items():
        if not idx:
            continue
        idx = np.asarray(idx)
        rows.append({
            "method": method,
            "column": col,
            "bin": lab,
            **_metrics(true[idx], pred[idx])
        })

    return rows


def benchmark_methods(df: pl.DataFrame, target_columns: list[str] = None, feature_columns: list[str] = None, n_gaps: int = 150, gap_lengths = None, n_bins: int = 4, seed: int = 7, gp_windows: int = 120, gp_length_scale: float = 30.0, gp_noise_level: float = 0.1, gp_optimize: bool = False, iter_max_iter: int = 10, plot_dir = None, debug_mode: bool = False, logging_info = None) -> pl.DataFrame:
    """
    Compara pchip, gp e iterative reconstruyendo huecos sintéticos.
    Inyecta huecos en posiciones válidas de cada columna objetivo, aplica los tres métodos y mide el error contra la verdad conocida, global y por bin de largo de hueco. Las mismas posiciones se usan para todos los métodos.
    """
    rng = np.random.default_rng(seed)

    if target_columns is None:
        target_columns = [c for c, dt in df.schema.items() if dt.is_float()]
    if feature_columns is None:
        feature_columns = [c for c, dt in df.schema.items() if dt.is_float()]

    if gap_lengths is None:
        gl = gap_length_distribution(df, columns = target_columns)
        gap_lengths = gl if gl.size > 0 else np.array([1, 2, 3, 5, 10, 20, 40])
    gap_lengths = np.asarray(gap_lengths)

    bins = bins_from_lengths(gap_lengths, n_bins = n_bins)
    labels = bin_labels(bins)

    injected = {}
    gapped_cols = []

    for col in target_columns:
        values = df[col].to_numpy().astype("float64")
        valid = np.isfinite(values)

        inj = inject_synthetic_gaps(valid, gap_lengths, n_gaps, rng)
        injected[col] = inj

        g = values.copy()
        g[inj] = np.nan
        gapped_cols.append(pl.Series(col, g.astype("float32")))

    gapped = df.with_columns(gapped_cols)

    if logging_info is not None:
        logging_info(debug_mode,
            f"Benchmark of interpolation methods (n_gaps: {n_gaps}, seed: {seed}, columns: {len(target_columns)})"
        )

    rows = []
    trues = {col: df[col].to_numpy() for col in target_columns}
    n_cols = len(target_columns)

    for i, col in enumerate(target_columns, start = 1):
        true = trues[col]

        if logging_info is not None:
            logging_info(debug_mode, f"{'pchip':>10}   |   {col:>18}   |   avance: {i / n_cols:6.1%}")
        rows += _scored_rows(
            "pchip", col, true,
            fill_pchip(gapped[col]).to_numpy(),
            injected[col], bins, labels
        )

        if logging_info is not None:
            logging_info(debug_mode, f"{'gp':>10}   |   {col:>18}   |   avance: {i / n_cols:6.1%}")
        rows += _scored_rows(
            "gp", col, true,
            fill_gaussian_process(
                gapped[col], windows = gp_windows, seed = seed,
                optimize_hyperparams = gp_optimize,
                length_scale = gp_length_scale, noise_level = gp_noise_level,
                only_mask = injected[col],
            )[0].to_numpy(),
            injected[col], bins, labels
        )

    if logging_info is not None:
        logging_info(debug_mode, f"{'iterative':>10}   |   MICE sobre {len(gapped):,} filas (max_iter: {iter_max_iter}, columnas: {len(feature_columns)})")
    imputed = fill_iterative(gapped, columns = feature_columns, seed = seed, max_iter = iter_max_iter)
    for col in target_columns:
        rows += _scored_rows(
            "iterative", col, trues[col],
            imputed[col].to_numpy(),
            injected[col], bins, labels
        )

    res = pl.DataFrame(rows)

    if plot_dir is not None:
        from ...figure.figure_interpolation import plot_benchmark_overall, plot_benchmark_by_window
        plot_benchmark_overall(res, save_path = plot_dir / "benchmark_overall.png")
        plot_benchmark_by_window(res, save_path = plot_dir / "benchmark_por_ventana.png")
        if logging_info is not None:
            logging_info(debug_mode, f"Benchmark plots saved in {plot_dir}")

    return res