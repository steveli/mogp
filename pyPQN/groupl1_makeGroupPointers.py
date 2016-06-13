import numpy as np


def groupl1_makeGroupPointers(groups):
    nVars = len(groups)
    nGroups = groups.max() + 1

    # First Count the Number of Elemets in each Group
    groupStart = np.empty(nGroups + 1, dtype=int)
    for i in xrange(nGroups):
        groupStart[i + 1] = (groups == i).sum()
    groupStart[0] = 0
    groupStart = np.cumsum(groupStart)
    # Now fill in the pointers to elements of the groups
    groupPtr = np.empty(nVars, dtype=int)
    groupPtr[:] = -1
    groupPtrInd = np.zeros(nGroups, dtype=int)
    for i in xrange(nVars):
        if groups[i] >= 0:
            grp = groups[i]
            groupPtr[groupStart[grp] + groupPtrInd[grp]] = i
            groupPtrInd[grp] += 1
    return groupStart, groupPtr

