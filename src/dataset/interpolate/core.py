import polars as pl
import numpy as np

def missing_mask(s: pl.Series) -> np.ndarray:
    """
    Máscara booleana de valores faltantes (null o NaN) de una Series.
    Unifica el criterio de faltante de todo el módulo, de modo que el resultado no depende de cómo se escribió el feather: pandas guarda los faltantes como null, Polars los guarda como NaN. Se consideran ambos.
    NaN solo se evalúa en columnas de punto flotante; para columnas no float (por ejemplo la columna Epoch, de tipo Datetime) se usa solo is_null, ya que is_nan no está definido sobre esos tipos. 
    inf NO se trata como faltante: los fill values ya se convierten a NaN en cdf_reader.
    """
    missing = s.is_null()
    if s.dtype.is_float():
        missing = missing | s.is_nan()
    return missing.to_numpy()


def nan_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """
    Detecta rachas consecutivas de True en una máscara booleana.
    Localiza los tramos contiguos de valores faltantes usando detección de flancos: se añade un False en cada extremo y se toma la diferencia discreta, de modo que un +1 marca el inicio de una racha y un -1 su fin. 
    Es la alternativa vectorizada a recorrer el arreglo elemento por elemento.
    """
    if mask.size == 0:
        return []

    padded = np.concatenate(([False], mask, [False]))
    diff = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1)

    return [(int(s), int(e - s)) for s, e in zip(starts, ends)]


def bin_labels(bins) -> list[str]:
    """
    Genera etiquetas de texto para intervalos definidos por sus bordes.
 
    Para bordes [1, 5, 20, inf] produce ["[1, 5)", "[5, 20)", "[20, inf)"]. 
    El último intervalo, si su borde superior es infinito, se rotula con inf.
    """
    labels = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        if np.isinf(hi):
            labels.append(f"[{int(lo)}, inf)")
        else:
            labels.append(f"[{int(lo)}, {int(hi)})")

    return labels


def assign_bin(length, bins) -> int:
    """
    Ubica un largo de hueco en el índice de intervalo que le corresponde.
    Búsqueda lineal sobre los bordes: devuelve el primer intervalo [lo, hi) que contiene el largo. Si ninguno lo contiene, devuelve el último.
    """
    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        if lo <= length < hi:
            return i

    return len(bins) - 2


def gap_length_distribution(df: pl.DataFrame, columns = None):
    """
    Reúne los largos de los huecos interiores de las columnas indicadas.
    Un hueco es interior si no toca el inicio ni el final de la serie; los huecos de borde se excluyen porque no son interpolables (no hay dato a ambos lados). 
    Sirve como insumo para dimensionar los intervalos de caracterización.
    """
    if columns is None:
        columns = list(df.columns)

    longs = []
    for col in columns:
        mask = missing_mask(df[col])
        n = mask.size
        for start, length in nan_runs(mask):
            if start == 0 or start + length == n:
                continue
            longs.append(length)

    return np.asarray(longs, dtype = int)


def auto_gap_bins(df, columns = None, n_bins = 4, bins = None):
    """
    Construye bordes de intervalo espaciados geométricamente para los huecos.
    Si se entregan bins, se respetan. Si no, se generan n_bins bordes entre 1 y el largo máximo observado usando escala geométrica, adecuada para una distribución con muchos huecos cortos y pocos largos (típico de series temporales). 
    Garantiza que el primer borde sea 1 y agrega inf al final.
    """
    if bins is not None:
        return bins

    longs = gap_length_distribution(df, columns)
    if longs.size == 0:
        return [1, np.inf]

    lmax = int(longs.max())
    if lmax <= 1:
        return [1, np.inf]
    
    pts = np.unique(np.round(np.geomspace(1, lmax, n_bins + 1)).astype(int))
    pts = pts[pts >= 1]
    if pts.size and pts[0] != 1:
        pts = np.insert(pts, 0, 1)

    border = [int(b) for b in pts[:n_bins]]
    if len(border) < 2:
        return [1, np.inf]

    return [*border, np.inf]


def characterize_gaps(df: pl.DataFrame, cfg: dict, debug_mode: bool, logging_info = None, bins = None) -> pl.DataFrame:
    """
    Resume, por columna, la estadística de huecos de datos faltantes. Para cada columna cuenta cuántos huecos interiores caen en cada intervalo de largo, cuántos son de borde, la fracción de faltantes y el largo máximo.
    Opcionalmente registra un resumen legible mediante logging_info.
    """
    bins = auto_gap_bins(df, bins = bins)
    labels = bin_labels(bins)

    rows = []
    for col in df.columns:
        mask = missing_mask(df[col])
        runs = nan_runs(mask)

        n = mask.size
        count = {lab: 0 for lab in labels}
        bord = 0
        max_long = 0

        for start, length in runs:
            max_long = max(max_long, length)
            in_bord = (start == 0) or (start + length == n)

            if in_bord:
                bord += 1
                continue
            count[labels[assign_bin(length, bins)]] += 1

        row = {
            "column": col,
            "nan_frac": float(mask.mean()),
            "n_gaps": len(runs),
            "bord": bord,
            "length_max_min": max_long,
        }

        row.update(count)
        rows.append(row)

    resume = pl.DataFrame(rows)

    if logging_info is not None:
        logging_info(debug_mode,
            f"Characterizaction of gaps (count per bin of minutes):"
        )
        for r in rows:
            logging_info(debug_mode,
                f"{r['column']:>18}   |   NaN: {r['nan_frac']:6.2%}   |   Gaps: {int(r['n_gaps']):5d}   |   Border: {int(r['bord']):4d}   |   Max Length: {int(r['length_max_min']):6d} min"
            )

    return resume



def gap_regions(df: pl.DataFrame, columns = None, min_long = 1) -> list:
    """
    Lista las regiones de huecos interiores interpolables, con sus tiempos.
    Para cada hueco interior de largo suficiente devuelve su columna, posición, largo y las marcas de tiempo de inicio y fin, tomadas de la columna Epoch.
    Es la entrada directa del paso de interpolación: indica qué tramos rellenar.
    """
    if columns is None:
        columns = list(df.columns)

    regions = []
    n = len(df)
    t = df["Epoch"].to_numpy()

    for col in columns:
        mask = missing_mask(df[col])
        for start, length in nan_runs(mask):
            if start == 0 or start + length == n:
                continue
            if length < min_long:
                continue

            regions.append({
                "column": col,
                "start": start,
                "length": length,
                "t_start": t[start],
                "t_end": t[start + length - 1]
            })

    return regions