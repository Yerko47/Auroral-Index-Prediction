import sys
import logging
from pathlib import Path

#* CONFIGURACION DE LOGGING
def config_logging(paths: dict) -> None:
    """
    Configura el sistema de logging con salida a archivo y consula
    """
    log_file = paths["log_file"]

    logging.basicConfig(
        level = logging.INFO,
        format = '%(asctime)s - %(levelname)s - %(message)s',
        handlers = [
            logging.FileHandler(log_file / "proceso.log", mode = "a", encoding = "utf-8"),
            logging.StreamHandler(sys.stdout)
        ]
    )


#* LOGGING INFO
def logging_info(debug_mode: bool, msg: str) -> None:
    if debug_mode:
        logging.info(msg)


#* TITULO DE LOGGING
def logging_titulo(debug_mode: bool, titulo: str, detalle: str = "") -> None:
    """
    Encabezado visible en el log, para separar bloques
    """
    if debug_mode:
        ancho = 70
        logging.info("="*ancho)
        logging.info(titulo.center(ancho))

        if detalle:
            logging.info(detalle.center(ancho))

        logging.info("="*ancho)