from pathlib import Path

def path_file() -> dict:
    """
    Diccionario con las carpetas necesarias para el funcionamiento del código.
    Si es que la carpeta no existe, se crea mediante este código
    """

    base = Path(Path(__file__).resolve().parent.parent.parent)

    project_paths = {
        "raw_file": ("data", "raw"),
        "processed_file": ("data", "processed"),
        "events_file": ("data", "events"),

        "model_result_file": ("results", "model"),
        "prediction_result_file": ("results", "prediction"),
        "metric_result_file": ("results", "metric"),

        "checkpoint_file": ("results", "checkpoints"),
    }

    paths = {key: base.joinpath(*parts) for key, parts in project_paths.items()}
    paths["config_file"] = base / "config"
    paths["config_model_file"] = base / "config" / "models"
    paths["log_file"] = base / "log"
    paths["figures"] = base / "figures"

    for path in paths.values():
        path.mkdir(parents = True, exist_ok = True)

    paths["base_file"] = base
    paths["omni_file"] = Path("/data/omni/hro_1min/")

    return paths


def create_new_file(paths: Path, name_file: str, new_name: str) -> Path:
    """
    Crear una nueva carpeta para guardar elementos puntuales que no son necesario tener en un diccionario
    """
    new_file = paths[name_file] / new_name
    
    new_file.mkdir(parents = True, exist_ok = True)

    return new_file
