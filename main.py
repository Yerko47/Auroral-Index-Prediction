from src.utils import (
    path_file, config_load, config_logging, set_seed,
    logging_titulo, logging_info, create_new_file,
)
from src.dataset import dataset
from src.dataset.apply import apply_interpolation, apply_scaling, save_interpolation, load_interpolation
from src.dataset.interpolate.core import characterize_gaps
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

    # 1b) Resumen de huecos/faltantes del dataset
    characterize_gaps(df, cfg, debug_mode, logging_info)

    # 2) Reuso: si ya existe el interpolado en data/processed, se carga y se omite benchmark + interpolacion
    cached = load_interpolation(cfg, paths, debug_mode, logging_info) if cfg["interpolation"]["reuse"] else None
    if cached is not None:
        df_interp, applied = cached
    else:
        # 2a) Benchmark (corre y grafica solo si cfg["interpolation"]["benchmark"] es True)
        bench = None
        if cfg["interpolation"]["benchmark"]:
            bench = benchmark_methods(df, cfg, paths, seed = seed,
                                      debug_mode = debug_mode, logging_info = logging_info)

        # 2b) Interpolacion segun config (devuelve datos SIN escalar + procedencia)
        df_interp, applied = apply_interpolation(df, cfg, benchmark_result = bench, seed = seed,
                                                 debug_mode = debug_mode, logging_info = logging_info)

        # 2c) Guardar interpolado (sin escalar) + procedencia en data/processed
        save_interpolation(df_interp, applied, cfg, paths, debug_mode, logging_info)

    # 4) Figuras ANTES de escalar (los graficos del benchmark se generan dentro de benchmark_methods)
    if cfg["project"]["plot"]:
        logging_info(debug_mode, "plot_history_gaps")
        plot_history_gaps(df_interp, df, applied,
                          save_dir = create_new_file(paths, "figures", "interpolation_gaps"))

    # 5) Escalado (scikit-learn); se guarda el scaler para desescalar despues
    df_scaled, scaler = apply_scaling(df_interp, cfg)

    logging_info(debug_mode, f"Interpolacion: {applied.height} zonas | escalado: {cfg['scaling']['method']}")

    return df_scaled, scaler


if __name__ == "__main__":
    main()