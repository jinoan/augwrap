import os
import random
import numpy as np

def set_random_seed(seed_num:int=42):
    os.environ["PYTHONHASHSEED"] = str(seed_num)
    random.seed(seed_num)
    np.random.seed(seed_num)