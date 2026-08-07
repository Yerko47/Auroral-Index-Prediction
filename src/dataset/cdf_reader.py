from pathlib import Path

import cdflib
import pandas as pd
import numpy as np

def cdf_read(cdf_file: Path, cfg: dict, debug_mode: bool, logging_info = None) -> pd.DataFrame:
    """
    """
    try:
        cdf = cdflib.CDF(cdf_file)
    except Exception as e:
        raise RuntimeError(f"Error opening or reading the CDF file {cdf_file}") from e

    cdf_rename = {"B_T": "F", "E": "E_Field"}

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
        values = np.asarray(arr, dtype = "float64")
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

    df = pd.DataFrame(data)
    df.index = pd.to_datetime(cdflib.cdfepoch.to_datetime(cdf["Epoch"][...]))
    df.index.name = "Epoch"
    df = df.sort_index()

    df = df.rename(columns = {native: name for name, native in cdf_rename.items()})

    if no_attrs:
        logging_info(debug_mode, f"Variables without FILLVAL/VALIDMIN/VALIDMAX: {', '.join(no_attrs)}")

        for col in df.columns:
            logging_info(debug_mode,
                f"{col:>18}  NaN: {df[col].isna().mean():6.2%} "
                f"min: {df[col].min():12.3f}    |    max: {df[col].max():12.3f}"
            )

    return df



