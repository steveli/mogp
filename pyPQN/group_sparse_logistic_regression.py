import numpy as np
import pylab as pl
from SoftmaxLoss2 import SoftmaxLoss2
from auxGroupLoss import auxGroupLoss
from groupl1_makeGroupPointers import groupl1_makeGroupPointers
from auxGroupL2Project import groupL2Proj
from minConF_PQN import minConf_PQN

import sys
if len(sys.argv) > 1:
    np.random.seed(int(sys.argv[1]))
else:
    np.random.seed(1)   # NOTE: check anomaly caused by np.random.seed(0)

# Generate synthetic data
nInstances = 100
nVars = 25
nClasses = 6
X = np.hstack((np.ones((nInstances, 1)),
               np.random.normal(size=(nInstances, nVars - 1))))
W = np.diag(np.random.uniform(size=nVars) > .75).dot(np.random.normal(size=(nVars, nClasses)))
y = np.argmax(X.dot(W) + np.random.normal(size=(nInstances, nClasses)), axis=1)

# Initial guess of parameters
W_groupSparse = np.zeros((nVars, nClasses - 1))

# Set up Objective Function
def funObj(W):
    return SoftmaxLoss2(W, X, y, nClasses)

# Set up Groups (don't penalized bias)
groups = np.vstack((np.ones((1, nClasses - 1), dtype=int) * -1,
                    np.outer(np.arange(nVars - 1),
                             np.ones(nClasses - 1, dtype=int))))
groups = np.ravel(groups)
nGroups = groups.max() + 1

# Initialize auxiliary variables that will bound norm
lambda_ = 10
alpha = np.zeros(nGroups)

def penalizedFunObj(W):
    return auxGroupLoss(W, groups, lambda_, funObj)

# Set up L_1,2 Projection Function
groupStart, groupPtr = groupl1_makeGroupPointers(groups)

def funProj(W):
    return groupL2Proj(W, nVars * (nClasses - 1), groupStart, groupPtr)

# Solve with PQN
print '\nComputing group-sparse multinomial logistic regression parameters...'
x0 = np.concatenate((np.ravel(W_groupSparse), alpha))
Walpha = minConf_PQN(penalizedFunObj,
        np.concatenate((np.ravel(W_groupSparse), alpha)), funProj,
        verbose=3)[0]

# Extract parameters from augmented vector
W_groupSparse = Walpha[0:nVars * (nClasses - 1)].reshape((nVars,nClasses - 1))
W_groupSparse[np.fabs(W_groupSparse) < 1e-4] = 0

print W_groupSparse

#print W_groupSparse

pl.subplot(1,2,1)
pl.imshow(W_groupSparse != 0, interpolation='nearest')
pl.gray()
pl.xticks([])
pl.yticks([])
pl.title('Sparsity Pattern')
pl.ylabel('variable')
pl.xlabel('class label')
pl.subplot(1, 2, 2)
pl.imshow(W_groupSparse, interpolation='nearest')
#pl.gray()
pl.xticks([])
pl.yticks([])
pl.title('Variable weights')
pl.ylabel('variable')
pl.xlabel('class label')

# Check selected variables
print 'Number of classes where bias variable was selected: %d (of %d)' % (
        np.count_nonzero(W_groupSparse[0]), nClasses -1)

for s in xrange(1, nVars):
    print 'Number of classes where variable %d was selected: %d (of %d)' % (
            s, np.count_nonzero(W_groupSparse[s]), nClasses - 1)

print 'Total number of variables selected: %d (of %d)' % (
        np.count_nonzero(W_groupSparse[1:].sum(axis=1)), nVars)

pl.show()

np.set_printoptions(formatter={'all': lambda x: '%10.4f' % x
                                                if ('%g' % x) != '0' else
                                                '%10s' % 0})
print W_groupSparse
print
