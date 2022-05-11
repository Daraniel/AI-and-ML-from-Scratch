import numpy as np


def rmse(x, y):
    return np.sqrt(np.mean((x - y) ** 2))
