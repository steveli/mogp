import numpy as np
import pylab as pl
from minConF_PQN import minConf_PQN
from projectRandom2 import randomProject
from SquaredError import SquaredError

import sys
if len(sys.argv) > 1:
    np.random.seed(int(sys.argv[1]))
else:
    np.random.seed(0)

# Generate Syntehtic Data
nInstances = 500
nVars = 50
X = np.random.normal(size=(nInstances, nVars))
w = np.random.normal(size=nVars) * (np.random.uniform(size=nVars) > .5)
y = X.dot(w) + np.random.normal(size=nInstances)

# Initial guess of parameters
wL1 = np.zeros(nVars)

# Set up Objective Function
def funObj(w):
    return SquaredError(w, X, y)

# Set up L1-Ball Projection
tau = 2
def funProj(w):
    return np.sign(w) * randomProject(np.fabs(w), tau)

# Solve with PQN
print '\nComputing optimal Lasso parameters...'
wL1 = minConf_PQN(funObj, wL1, funProj)[0]
wL1[np.fabs(wL1) < 1e-4] = 0

# Check sparsity of solution
np.set_printoptions(formatter={'all': lambda x: '%8.4f' % x
                                                if ('%g' % x) != '0' else
                                                '%8s' % 0})
print wL1
print 'Number of non-zero variables in solution: %d (of %d)' % (
        np.count_nonzero(wL1), len(wL1))

pl.subplot(1, 2, 1)
#imagesc(wL1);colormap gray
pl.imshow(np.outer(wL1, np.ones(10)), interpolation='none')
pl.gray()
pl.xticks([])
pl.yticks([])
pl.title(' Weights')
pl.subplot(1, 2, 2)
#imagesc(wL1~=0);colormap gray
pl.imshow(np.outer(wL1 != 0, np.ones(10)), interpolation='none')
pl.gray()
pl.xticks([])
pl.yticks([])
pl.title('Sparsity of wL1')
pl.show()
