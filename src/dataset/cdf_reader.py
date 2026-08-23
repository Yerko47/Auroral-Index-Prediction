from datetime import datetime
from pathlib import Path

import cdflib
import polars as pl
import numpy as np

def cdf_reader(cdf_file: Path, cfg: dict, debug_mode: bool, logging_info = None) -> pl.DataFrame:
    """
    """
    try:
        cdf = cdflib.CDF(cdf_file)
    except Exception as e:
        raise RuntimeError(f"Error opening or reading the CDF file {cdf_file}") from e

    cdf_rename = {
        "B_T": "F"
    }

    columns_config = cfg["dataset"]["omni_list"] + cfg["dataset"]["auroral_index"]
    columns_dataset = [cdf_rename.get(col, col) for col in columns_config]

    available = set(cdf.cdf_info().zVariables)

    missing = [c for c in ["Epoch", *columns_dataset] if c not in available]
    if missing:
        raise KeyError(f"Variables missing in {cdf_file}: {missing}")

    data: dict[str, np.ndarray] = {}
    no_attrs: list[str] = []


    for var in columns_dataset:
        arr = cdf[var][...]
        values = np.asarray(arr, dtype = "float32")
        finite = np.isfinite(values)
        mask = ~finite

        atts = cdf.varattsget(var)
        has_criteria = False

        if "FILLVAL" in atts:
            fillval = float(np.ravel(atts["FILLVAL"])[0])
            mask |= finite & np.isclose(values, fillval)
            has_criteria = True

        if "VALIDMIN" in atts:
            lo = float(np.ravel(atts["VALIDMIN"])[0])
            mask |= finite & (values < lo)
            has_criteria = True

        if "VALIDMAX" in atts:
            hi = float(np.ravel(atts["VALIDMAX"])[0])
            mask |= finite & (values > hi)
            has_criteria = True

        if not has_criteria:
            no_attrs.append(var)

        values[mask] = np.nan
        data[var] = values

    epoch = cdflib.cdfepoch.to_datetime(cdf["Epoch"][...])
    df = pl.DataFrame(data)
    df = df.with_columns(pl.Series("Epoch", epoch))
    df = df.sort("Epoch")

    df = df.rename(mapping = {native: name for name, native in cdf_rename.items()})

    if no_attrs and logging_info is not None:
        logging_info(debug_mode, f"Variables without FILLVAL/VALIDMIN/VALIDMAX: {', '.join(no_attrs)}")

        for col in [c for c, dt in df.schema.items() if dt.is_float()]:
            logging_info(debug_mode,
                f"{col:>18} NaN: {df[col].is_null().mean():6.2%} "
                f"min: {df[col].min():12.3f}   |   max: {df[col].max():12.3f}"
            )

    return df


def dataset(cfg: dict, paths: dict, debug_mode: bool, logging_info = None) -> pl.DataFrame:
    """
    """
    start_time = datetime.fromisoformat(cfg["dataset"]["time_range"]["start"])
    end_time = datetime.fromisoformat(cfg["dataset"]["time_range"]["end"])

    save_feather_file = paths["raw_file"] / f"data_{start_time.year}_to_{end_time.year}.feather"

    if save_feather_file.exists():
        logging_info(debug_mode,
            f"Loading the raw data from feather file into {save_feather_file}"
        )
        df = pl.read_ipc(save_feather_file)

        return df

    logging_info(debug_mode,
        f"Reading OMNI data from date {start_time.strftime('%Y-%m-%d')} to {end_time.strftime('%Y-%m-%d')}\n"
        + "=" * 70
    )

    omni_path = paths["omni_file"]
    date_array = pl.date_range(start_time, end_time, interval = "1mo", eager = True)

    o = []
    for date in date_array:
        name_file = f"omni_hro_1min_{date.strftime('%Y%m%d')}_v01.cdf"
        cdf = cdf_reader(cdf_file = omni_path / f"{date.year}" / name_file, cfg = cfg, debug_mode = debug_mode, logging_info = logging_info)
        logging_info(debug_mode,
            f"The file {name_file} is loading"
        )
        o.append(cdf)

    df = pl.concat(o)
    df = df.sort("Epoch")

    df.write_ipc(save_feather_file)
    logging_info(debug_mode,
        f"File save in {save_feather_file}\n"
        + "=" * 70
    )

    return df
