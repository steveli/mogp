from __future__ import division
import numpy as np


def randomProject(c, lambda_):
    # Finds solution p of:
    #   min_p ||c - p||_2
    #    s.t. |p| <= lambda
    #
    # Assumes all elements of c are positive (this version handles ties)
    #
    # This version operates in-place
    # (slower in Matlab, but faster in C)

    nVars = len(c)

    if c.sum() <= lambda_:
        p = c
        return p

    mink = 1
    p = c[c > 0]
    nVars = len(p)
    maxk = nVars
    offset = 0
    while True:

        # Chose a (nearly) random element of the partition
        # (we take the median of 3 random elements to help
        #   protect us against choosing a bad pivot)
        cand1 = p[mink - 1 +
                  int(np.ceil(np.random.uniform() * (maxk - mink + 1))) - 1]
        cand2 = p[mink - 1 +
                  int(np.ceil(np.random.uniform() * (maxk - mink + 1))) - 1]
        cand3 = p[mink - 1 +
                  int(np.ceil(np.random.uniform() * (maxk - mink + 1))) - 1]
        p_k = np.median([cand1, cand2, cand3])

        # Partition Elements in range {mink:maxk} around p_k
        lowerLen = 0
        middleLen = 0
        for i in xrange(mink, maxk + 1):
            if p[i - 1] > p_k:
                p[i - 1], p[mink + lowerLen - 1] = p[mink + lowerLen - 1], p[i - 1]
                lowerLen += 1
                if p[i - 1] == p_k:
                    p[i - 1], p[mink + middleLen - 1] = p[mink + middleLen - 1], p[i - 1]
                middleLen += 1
            elif p[i - 1] == p_k:
                p[i - 1], p[mink + middleLen - 1] = p[mink + middleLen - 1], p[i - 1]
                middleLen += 1

        middleLen = middleLen - lowerLen
        upperLen = maxk - mink - lowerLen-middleLen + 1

        # Find out what k value this element corresponds to
        k = lowerLen + middleLen + mink - 1

        # Compute running sum from 1 up to k-1
        s1 = offset + sum(p[(mink - 1):(mink + lowerLen - 1)]) + p_k * (middleLen - 1)

        # Compute Soft-Threshold up to k
        LHS = s1 - (k - 1) * p_k

        if k < nVars:
            # Find element k+1
            if upperLen == 0:
                p_kP1 = p_maxkP1
            else:
                p_kP1 = max(p[(mink + lowerLen + middleLen - 1):(maxk + 1)])
        else:
            # We pad the end of the array with an extra '0' element
            p_kP1 = 0

        # Compute Soft-Threshold up to k+1
        s2 = s1 + p_k
        RHS = s2 - k * p_kP1

        if lambda_ >= LHS and (lambda_ < RHS or upperLen == 0):
            break

        if lambda_ < LHS: # Decrease maxk
            maxk = k - middleLen
            p_maxkP1 = p_k
        else: # lambda > RHS, Increase mink
            mink = k + 1
            offset = s2

    tau = p_k - (lambda_ - LHS) / k
    p = c - tau
    p[p < 0] = 0

    return p
