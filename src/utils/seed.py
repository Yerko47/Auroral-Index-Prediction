import os
import random

import numpy as np
import torch

#* SET SEED
def set_seed(seed: int = 42) -> None:
    """
    Fija la semilla en todas las fuentes de aleatoridad para reproducibilidad

    Params:
        seed (int): Semilla a usar
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False

#* SEED WORKER
def seed_worker(worker_id: int) -> None:
    """
    Inicializador de semilla por worker del DataLoader.
    Se utiliza de la siguiente forma:
        g = torch.Genereator()
        g.manual_seed(42)
        DataLoader(..., worker_init_fn = seed_worker, generator = g)
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

