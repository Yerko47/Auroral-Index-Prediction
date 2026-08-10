import numpy as np
import pandas as pd


def nan_runs(mask):
    """
    """
    if mask.size == 0:
        return []

    padded = np.concatenate(([False], mask, [False]))
    diff = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1)

    return [(int(s), int(e - s)) for s, e in zip(starts, ends)]


def bin_labels(bins):
    """
    """
    labels = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        if np.isinf(hi):
            labels.append(f"[{int(lo)}], inf)")
        else:
            labels.append(f"[{int(lo)}], {int(hi)})")
    return labels


def assign_bin(length, bins):
    """
    """
    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        if lo <= length < hi:
            return i
    return len(bins) - 2


def gap_length_distribution(df, columns = None):
    """
    """
    if columns is None:
        columns = list(df.columns)
    longs = []

    for col in columns:
        mask = df[col].isna().to_numpy()
        n = mask.size
        for start, length in nan_runs(mask):
            if start == 0 or start + length == n:
                continue
            longs.append(length)

    return np.asarray(longs, dtype = int)


def auto_gap_bins(df, columns = None, n_bins = 4, bins = None):
    """
    """
    if bins is not None: return bins

    longs = gap_length_distribution(df, columns)
    if longs.size == 0: return [1, np.inf]

    lmax = int(longs.max())
    if lmax <= 1: return [1, np.inf]

    pts = np.unique(np.round(np.geomspace(1, lmax, n_bins +1)).astype(int))
    pts = pts[pts >= 1]
    if pts.size and pts[0] != 1:
        pts = np.insert(pts, 0 , 1)

    border = [int(b) for b in pts[:n_bins]]
    if len(border) < 2: return [1, np.inf]

    return [*border, np.inf]


def characterize_gaps(df, cfg, debug_mode, logging_info = None, bins = None):
    """
    """
    bins = auto_gap_bins(df, bins = bins)
    labels = bin_labels(bins)

    rows = []
    for col in df.columns:
        mask = df[col].isna().to_numpy()
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
            "length_max_min": max_long
        }
        row.update(count)
        rows.append(row)

    resume = pd.DataFrame(rows).set_index("column")

    if logging_info is not None:
        logging_info(debug_mode,"Characterization of gaps (count per bin of minutes):")
        for col, r in resume.iterrows():
            logging_info(debug_mode,
                         f"{col:>18} | NaN: {r['nan_frac']:6.2%} | GAPS: {int(r['n_gaps']):5d} | BORDER: {int(r['bord']):4d} | MAX LENGTH: {int(r['length_max_min']):6d} min"
                         )

    return resume


def gap_regions(df, columns = None, min_long = 1):
    """
    """
    if columns is None:
        columns = list(df.columns)
    regions = []
    n = len(df)

    for col in columns:
        valid = np.isfinite(df[col].to_numpy())
        for start, length in nan_runs(~valid):
            if start == 0 or start + length == n: continue
            if length < min_long: continue
            regions.append({
                "column": col,
                "start": start,
                "length": length,
                "t_start": df.index[start],
                "t_end": df.index[start + length -1]
            })
    return regions