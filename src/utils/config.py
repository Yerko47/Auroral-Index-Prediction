import yaml
from pathlib import Path

def config_load(paths: dict) -> dict:
    """
    Diccionario con todas las variables a utilizar para la predicción
    """
    config_file = paths["config_file"] / "config.yaml"
    
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    type_model = config["model"]["type"].upper()

    model_config_file = paths["config_model_file"] / f"{type_model}.yaml"

    with open(model_config_file, "r") as f:
        config_model = yaml.safe_load(f)
    
    config["model"]["parameters"] = config_model

    return config
