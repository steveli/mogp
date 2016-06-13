import numpy as np
from minConF_PQN import minConf_PQN
from projectSimplex import projectSimplex
from SquaredError import SquaredError


# Generate Syntehtic Data
nInstances = 50
nVars = 10
X = np.random.normal(size=(nInstances, nVars))
w = np.random.uniform(size=nVars) * (np.random.uniform(size=nVars) > .5)
y = X.dot(w) + np.random.normal(size=nInstances)

# Initial guess of parameters
wSimplex = np.zeros(nVars)

# Set up Objective Function
def funObj(w):
    return SquaredError(w, X, y)

# Set up Simplex Projection Function
def funProj(w):
    return projectSimplex(w)

# Solve with PQN
print 'Computing optimal linear regression parameters on the simplex...'
wSimplex = minConf_PQN(funObj, wSimplex, funProj)[0]

# Check if variable lie in simplex
print wSimplex
print 'Min value of wSimplex: %.3f' % min(wSimplex)
print 'Max value of wSimplex: %.3f' % max(wSimplex)
print 'Sum of wSimplex: %.3f' % sum(wSimplex)

