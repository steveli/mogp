from __future__ import division
import numpy as np


def projectSimplex(v):
# Compute the minimum L2-distance projection of vector v onto the probability simplex
    nVars = len(v)
    mu = np.sort(v)
    mu = mu[::-1]
    sm = 0
    for j in xrange(nVars):
        sm = sm + mu[j]
        if mu[j] - (1 / (j + 1)) * (sm - 1) > 0:
            row = j + 1
            sm_row = sm
    theta = (1 / row) * (sm_row - 1)
    w = v - theta
    w[w < 0] = 0
    return w
