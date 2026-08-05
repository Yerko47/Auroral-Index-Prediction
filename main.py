import pandas as pd

from src.utils import *
#from src.dataset.interpolation import run_interpolation


def main():
    paths = path_file()
    cfg = config_load(paths)
    config_logging(paths)

    debug_model = cfg["project"]["logging"]

    set_seed(seed = cfg["project"]["seed"])

    logging_titulo(debug_model, titulo = f"{cfg['project']['name']}", detalle = f"Autor: {cfg['project']['author']}    |    versión: {cfg['project']['version']}")

    #df = pd.read_feather(paths["raw_file"] / "omni_data_1995_to_2018.feather")
    #print(df.isna().sum())
    #* df debe venir de la lectura de datos (cdf_read); el relleno opera sobre el
    #* DataFrame con los fill values ya enmascarados a NaN
    # resultado = run_interpolation(df, cfg, paths, debug_model, logging_info)
    # df_relleno, flags = resultado["relleno"], resultado["flags"]



if __name__ == "__main__":
    main()