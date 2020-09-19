import os
import random
import numpy as np
from tensorflow.random import set_seed
SEED = 42

def setup_seed():
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    set_seed(SEED)
    return