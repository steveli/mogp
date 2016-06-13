import numpy as np


def isLegal(v):
    return (np.all(np.isreal(v)) and
            not np.any(np.isnan(v)) and
            not np.any(np.isinf(v)))
