import numpy as np
import polars as pl

import matplotlib
matplotlib.use("Agg")   # backend sin ventana; guarda a archivo
import matplotlib.pyplot as plt

_METHOD_COLORS = {"pchip": "#004084", "gp": "#7A0000", "iterative": "#0D8400"}


def plot_benchmark_overall(res, metric="rmse", save_path=None):
    """
    Barras de la métrica agregada (bin='all') por variable y método.
    """
    d = res.filter(pl.col("bin") == "all")
    cols = d["column"].unique(maintain_order=True).to_list()
    methods = d["method"].unique(maintain_order=True).to_list()

    fig, ax = plt.subplots(figsize=(max(6, 1.5 * len(cols)), 4))
    w = 0.8 / len(methods)
    x = np.arange(len(cols))
    for i, m in enumerate(methods):
        vals = []
        for c in cols:
            sub = d.filter((pl.col("column") == c) & (pl.col("method") == m))
            vals.append(sub[metric][0] if sub.height else np.nan)
        ax.bar(x + i * w, vals, w, label=m, color=_METHOD_COLORS.get(m))

    ax.set_xticks(x + w * (len(methods) - 1) / 2)
    ax.set_xticklabels(cols)
    ax.set_ylabel(metric.upper())
    ax.set_title(f"Benchmark: {metric.upper()} por variable (todos los gaps)")
    ax.legend(title="metodo")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130)
        plt.close(fig)
    return fig


def plot_benchmark_by_window(res, metric="rmse", save_path=None):
    """
    Barras de la métrica por bin de largo de gap, un subplot por variable.
    """
    d = res.filter(pl.col("bin") != "all")
    cols = d["column"].unique(maintain_order=True).to_list()
    methods = d["method"].unique(maintain_order=True).to_list()
    bins = d["bin"].unique(maintain_order=True).to_list()

    fig, axes = plt.subplots(1, len(cols), figsize=(4.5 * len(cols), 4), squeeze=False)
    x = np.arange(len(bins))
    w = 0.8 / len(methods)
    for j, c in enumerate(cols):
        ax = axes[0][j]
        for i, m in enumerate(methods):
            vals = []
            for b in bins:
                sub = d.filter((pl.col("column") == c) & (pl.col("method") == m) & (pl.col("bin") == b))
                vals.append(sub[metric][0] if sub.height else np.nan)
            ax.bar(x + i * w, vals, w, label=m, color=_METHOD_COLORS.get(m))
        ax.set_xticks(x + w * (len(methods) - 1) / 2)
        ax.set_xticklabels(bins, rotation=30, ha="right")
        ax.set_title(c)
        ax.set_xlabel("ventana (largo de gap, min)")
        if j == 0:
            ax.set_ylabel(metric.upper())
    axes[0][-1].legend(title="metodo")
    fig.suptitle(f"Benchmark: {metric.upper()} por ventana de interpolacion")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130)
        plt.close(fig)
    return fig


def plot_history_gaps(df_interp, df_original, applied, save_dir, context=200, max_figs=50):
    """
    Una figura por zona interpolada (antes del escalado): la variable afectada con el tramo relleno resaltado, más las demás variables como contexto.
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    if applied.height == 0:
        return []

    t = df_interp["Epoch"].to_numpy() if "Epoch" in df_interp.columns else np.arange(len(df_interp))
    variables = [c for c, dt in df_interp.schema.items() if dt.is_float()]
    n = len(df_interp)
    paths = []

    for k, row in enumerate(applied.iter_rows(named=True)):
        if k >= max_figs:
            break
        start, length = row["start"], row["length"]
        col, m = row["column"], row["method"]
        lo = max(0, start - context)
        hi = min(n, start + length + context)

        fig, axes = plt.subplots(len(variables), 1, figsize=(10, 1.8 * len(variables)),
                                 sharex=True, squeeze=False)
        seg = slice(start - lo, start - lo + length)
        color = _METHOD_COLORS.get(m, "#d62728")
        for ax, v in zip(axes[:, 0], variables):
            tt = t[lo:hi]
            ax.plot(tt, df_original[v].to_numpy()[lo:hi], color="#333333", lw=0.9, label="original")
            ax.axvspan(tt[seg][0], tt[seg][-1], color=color, alpha=0.12)   # zona de interpolacion en todas las variables
            if v == col:
                interp = df_interp[v].to_numpy()[lo:hi]
                ax.plot(tt[seg], interp[seg], color=color, lw=1.8, label=f"interp ({m})")
            ax.set_ylabel(v)
            ax.legend(loc="upper right", fontsize=7)
        axes[0, 0].set_title(f"Zona interpolada en {col} | metodo={m} | largo={length} min")
        axes[-1, 0].set_xlabel("tiempo")
        fig.tight_layout()

        p = f"{save_dir}/gap_{k:03d}_{col}_{m}.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        paths.append(p)
    return paths