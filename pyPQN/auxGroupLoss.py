import numpy as np


def auxGroupLoss(w, groups, lambda_, funObj):
    p = len(groups)
    nGroups = len(w) - p
    f, g = funObj(w[:p])

    f = f + sum(lambda_ * w[p:])
    g = np.concatenate((g, lambda_ * np.ones(nGroups)))

    return f, g
