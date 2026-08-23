from src.utils import (
    path_file, config_load, config_logging, set_seed,
    logging_titulo, logging_info, create_new_file,
)
from src.dataset import dataset
from src.dataset.apply import apply_interpolation, apply_scaling, save_interpolation
from src.dataset.interpolate.benchmark import benchmark_methods
from src.figure.figure_interpolation import plot_history_gaps


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

    # 2) Benchmark (solo si el metodo lo necesita para elegir ganador: best)
    bench = None
    if icfg["method"] == "best":
        gcfg = icfg.get("gp", {})
        itcfg = icfg.get("iterative", {})
        plot_dir = (create_new_file(paths, "figures", "interpolation")
                    if cfg["project"].get("plot", False) and icfg.get("plot_benchmark", False)
                    else None)
        bench = benchmark_methods(
            df, seed=seed,
            n_gaps=icfg.get("n_gaps_por_repeticion", 40),
            n_bins=icfg.get("n_bins", 4),
            gp_windows=gcfg.get("windows", 120),
            gp_length_scale=gcfg.get("length_scale", 30.0),
            gp_noise_level=gcfg.get("noise_level", 0.1),
            gp_optimize=gcfg.get("optimize_hyperparams", False),
            iter_max_iter=itcfg.get("max_iter", 10),
            plot_dir=plot_dir,
            debug_mode=debug_mode, logging_info=logging_info,
        )

    # 3) Interpolacion segun config (devuelve datos SIN escalar + procedencia)
    df_interp, applied = apply_interpolation(df, cfg, benchmark_result=bench, seed=seed,
                                             debug_mode=debug_mode, logging_info=logging_info)

    # 3b) Guardar interpolado (sin escalar) + procedencia en data/processed
    save_interpolation(df_interp, applied, cfg, paths, debug_mode, logging_info)

    # 4) Figuras ANTES de escalar (los graficos del benchmark se generan dentro de benchmark_methods)
    if cfg["project"].get("plot", False):
        if icfg.get("plot_history_diff", False):
            logging_info(debug_mode, "plot_history_gaps")
            plot_history_gaps(df_interp, df, applied,
                              save_dir=create_new_file(paths, "figures", "interpolation_gaps"),
                              max_figs=icfg.get("max_figuras", 50))

    # 5) Escalado (scikit-learn); se guarda el scaler para desescalar despues
    df_scaled, scaler = apply_scaling(df_interp, cfg)

    logging_info(debug_mode, f"Interpolacion: {applied.height} zonas | escalado: {cfg['scaling']['method']}")

    return df_scaled, scaler


if __name__ == "__main__":
    main()