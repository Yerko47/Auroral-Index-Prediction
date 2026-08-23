import polars as pl

from src.utils import *
from src.dataset import *
from src.dataset.apply import apply_interpolation, apply_scaling
from src.dataset.interpolate.benchmark import benchmark_methods
from src.figure.figure_interpolation import (
    plot_benchmark_overall, plot_benchmark_by_window, plot_history_gaps,
)


def main():
    paths = path_file()
    cfg = config_load(paths)
    config_logging(paths)

    debug_mode = cfg["project"]["logging"]
    seed = cfg["project"]["seed"]
    set_seed(seed=seed)

    logging_titulo(debug_mode, titulo=f"{cfg['project']['name']}",
                   detalle=f"Autor: {cfg['project']['author']}    |    versión: {cfg['project']['version']}")

    # 1) Datos crudos
    df = dataset(cfg, paths, debug_mode, logging_info)

    icfg = cfg["interpolation"]

    # 2) Benchmark (si esta activado o si el metodo es "best")
    bench = None
    if icfg.get("benchmark", False) or icfg["method"] == "best":
        logging_info(debug_mode, "benchmarck")
        bench = benchmark_methods(df, seed=seed)

    # 3) Interpolacion segun config (devuelve datos SIN escalar + procedencia)
    logging_info(debug_mode, "interpolation")
    df_interp, applied = apply_interpolation(df, cfg, benchmark_result=bench, seed=seed)

    # 4) Figuras ANTES de escalar
    if cfg["project"].get("plot", False):
        if icfg.get("plot_benchmark", False) and bench is not None:
            logging_info(debug_mode, "plot_1")
            plot_benchmark_overall(bench, save_path=create_new_file(paths, "figure", "interpolation") / "benchmark_overall.png")
            plot_benchmark_by_window(bench, save_path=(paths, "figure", "interpolation")/ "benchmark_por_ventana.png")
        if icfg.get("plot_history_diff", False):
            logging_info(debug_mode, "plot_2")
            plot_history_gaps(df_interp, df, applied,
                              save_dir=(paths, "figure", "interpolation_gaps"),
                              max_figs=icfg.get("max_figuras", 50))

    # 5) Escalado (scikit-learn); se guarda el scaler para desescalar despues
    df_scaled, scaler = apply_scaling(df_interp, cfg)

    logging_info(debug_mode, f"Interpolacion: {applied.height} zonas | escalado: {cfg['scaling']['method']}")

    return df_scaled, scaler


if __name__ == "__main__":
    main()