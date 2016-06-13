from __future__ import division
import numpy as np


# Function to solve the projection for a single group
#def projectAux(w, alpha):
#    nw = np.linalg.norm(w)
#    if nw > alpha:
#        avg = (nw + alpha) / 2
#        if avg < 0:
#            w[:] = 0
#            alpha = 0
#        else:
#            w = w * avg / nw
#            alpha = avg
#    return w, alpha


def projectAux(w, alpha):
    nw = np.linalg.norm(w)
    if nw <= -alpha:
        w[:] = 0
        alpha = 0
    elif nw >= alpha:
        scale = 0.5 * (1 + alpha / nw)
        w = scale * w
        alpha = scale * nw
    return w, alpha


def groupL2Proj(w, p, groupStart, groupPtr):
    alpha = w[p:].copy()
    w = w[:p].copy()

    for i in xrange(len(groupStart) - 1):
        groupInd = groupPtr[groupStart[i]:groupStart[i + 1]]
        w[groupInd], alpha[i] = projectAux(w[groupInd], alpha[i])

    return np.concatenate((w, alpha))
