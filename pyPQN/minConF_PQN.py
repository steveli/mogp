from __future__ import division
import numpy as np
from numpy.linalg import norm, solve
from isLegal import isLegal
from polyinterp import polyinterp
from minConF_SPG import minConF_SPG


def lbfgsUpdate(y, s, corrections, debug, old_dirs, old_stps, Hdiag):
    if y.dot(s) > 1e-10:
        numVars, numCorrections = old_dirs.shape
        if numCorrections < corrections:
            # Full Update
            new_dirs = np.empty((numVars, numCorrections + 1))
            new_stps = np.empty((numVars, numCorrections + 1))
            new_dirs[:, :-1] = old_dirs
            new_stps[:, :-1] = old_stps
        else:
            # Limited-Memory Update
            new_dirs = np.empty((numVars, corrections))
            new_stps = np.empty((numVars, corrections))
            new_dirs[:, :-1] = old_dirs[:, 1:]
            new_stps[:, :-1] = old_stps[:, 1:]
        new_dirs[:, -1] = s
        new_stps[:, -1] = y

        # Update scale of initial Hessian approximation
        Hdiag = y.dot(s) / y.dot(y)
    else:
        if debug:
            print 'Skipping Update'
        new_dirs = old_dirs
        new_stps = old_stps

    return new_dirs, new_stps, Hdiag


def minConf_PQN(funObj, x, funProj,
                verbose=2,
                optTol=1e-6,
                maxIter=500,
                maxProject=100000,
                numDiff=0,
                suffDec=1e-4,
                corrections=10,
                adjustStep=0,
                bbInit=1,
                SPGoptTol=1e-6,
                SPGiters=10,
                SPGtestOpt=0):
    """Function for using a limited-memory projected quasi-Newton to solve
    problems of the form:

    min funObj(x) s.t. x in C

    The projected quasi-Newton sub-problems are solved the spectral projected
    gradient algorithm

    @funObj(x):
        function to minimize (returns gradient as second argument)
    @funProj(x):
        function that returns projection of x onto C

    options:

    verbose:
        level of verbosity
        (0: no output, 1: final, 2: iter (default), 3: debug)
    optTol:
        tolerance used to check for progress (default: 1e-6)
    maxIter:
        maximum number of calls to funObj (default: 500)
    maxProject:
        maximum number of calls to funProj (default: 100000)
    numDiff:
        compute derivatives numerically
        (0: use user-supplied derivatives (default),
         1: use finite differences,
         2: use complex differentials)
    suffDec:
        sufficient decrease parameter in Armijo condition (default: 1e-4)
    corrections:
        number of lbfgs corrections to store (default: 10)
    adjustStep:
        use quadratic initialization of line search (default: 0)
    bbInit:
        initialize sub-problem with Barzilai-Borwein step (default: 1)
    SPGoptTol:
        optimality tolerance for SPG direction finding (default: 1e-6)
    SPGiters:
        maximum number of iterations for SPG direction finding (default: 10)
    """

    nVars = len(x)

    # Output Parameter Settings
    if verbose >= 3:
        print 'Running PQN...'
        print 'Number of L-BFGS Corrections to store: %d' % corrections
        print 'Spectral initialization of SPG: %d' % bbInit
        print 'Maximum number of SPG iterations: %d' % SPGiters
        print 'SPG optimality tolerance: %.2e' % SPGoptTol
        print 'PQN optimality tolerance: %.2e' % optTol
        print 'Quadratic initialization of line search: %d' % adjustStep
        print 'Maximum number of function evaluations: %d' % maxIter
        print 'Maximum number of projections: %d' % maxProject

    # Output Log
    if verbose >= 2:
        print '%10s %10s %10s %15s %15s %15s' % (
                'Iteration', 'FunEvals', 'Projection',
                'Step Length', 'Function Val', 'Opt Cond')

    # Make objective function (if using numerical derivatives)
    funEvalMultiplier = 1
    # FIXME implement autoGrad
    #if numDiff:
    #    if numDiff == 2:
    #        useComplex = 1
    #    else:
    #        useComplex = 0
    #    funObj = @(x)autoGrad(x,useComplex,funObj)
    #    funEvalMultiplier = nVars+1-useComplex

    # Project initial parameter vector
    x = funProj(x)
    projects = 1

    # Evaluate initial parameters
    f, g = funObj(x)
    funEvals = 1

    # Check Optimality of Initial Point
    projects += 1

    if norm(funProj(x - g) - x, 1) < optTol:
        if verbose >= 1:
            print 'First-Order Optimality Conditions Below optTol at Initial Point'
        return x, f, funEvals

    f_old, g_old, x_old = 0, 0, 0
    i = 1

    while funEvals <= maxIter:

        # Compute Step Direction
        if i == 1:
            p = funProj(x - g)
            projects += 1
            S = np.zeros((nVars, 0))
            Y = np.zeros((nVars, 0))
            Hdiag = 1
        else:
            y = g - g_old
            s = x - x_old
            S, Y, Hdiag = lbfgsUpdate(y, s, corrections, verbose==3,
                                      S, Y, Hdiag)

            # Make Compact Representation
            k = Y.shape[1]
            L = np.zeros((k, k))
            for j in xrange(k):
                L[(j + 1):, j] = S[:, (j + 1):].T.dot(Y[:, j])

            N = np.hstack((S / Hdiag, Y))
            #M = np.vstack((np.hstack((S.T.dot(S) / Hdiag, L)),
            #               np.hstack((L.T, -np.diag(np.diag(S.T.dot(Y)))))))
            M = np.empty((k * 2, k * 2))
            M[:k, :k] = S.T.dot(S) / Hdiag
            M[:k, k:] = L
            M[k:, :k] = L.T
            M[k:, k:] = -np.diag(np.diag(S.T.dot(Y)))

            def HvFunc(v):
                return v / Hdiag - N.dot(solve(M, (N.T.dot(v))))

            if bbInit:
                # Use Barzilai-Borwein step to initialize sub-problem
                alpha = s.dot(s) / s.dot(y)
                if alpha <= 1e-10 or alpha > 1e10:
                    alpha = 1 / norm(g)

                # Solve Sub-problem
                # FIXME
                #xSubInit = x - alpha * g
                feasibleInit = 0
            else:
                # FIXME
                #xSubInit = x
                feasibleInit = 1

            # Solve Sub-problem
            p, subProjects = solveSubProblem(x, g, HvFunc, funProj, SPGoptTol,
                                             SPGiters, SPGtestOpt,
                                             feasibleInit, x)
            projects += subProjects

        d = p - x
        g_old = g
        x_old = x

        # Check that Progress can be made along the direction
        gtd = g.dot(d)
        if gtd > -optTol:
            if verbose >= 1:
                print 'Directional Derivative below optTol'
            break

        # Select Initial Guess to step length
        if i == 1 or adjustStep == 0:
            t = 1
        else:
            t = min(1, 2 * (f - f_old) / gtd)

        # Bound Step length on first iteration
        if i == 1:
            t = min(1, 1 / norm(g, 1))

        # Evaluate the Objective and Gradient at the Initial Step
        x_new = x + t * d
        f_new, g_new = funObj(x_new)
        funEvals += 1

        # Backtracking Line Search
        f_old = f
        while f_new > f + suffDec * t * gtd or not isLegal(f_new):
            temp = t

            # Backtrack to next trial value
            if not (isLegal(f_new) and isLegal(g_new)):
                if verbose == 3:
                    print 'Halving Step Size'
                t /= 2
            else:
                if verbose == 3:
                    print 'Cubic Backtracking'
                t = polyinterp(np.array([[0, f, gtd],
                                         [t, f_new, g_new.dot(d)]]))

            # Adjust if change is too small/large
            if t < temp * 1e-3:
                if verbose == 3:
                    print 'Interpolated value too small, Adjusting'
                t = temp * 1e-3
            elif t > temp * 0.6:
                if verbose == 3:
                    print 'Interpolated value too large, Adjusting'
                t = temp * 0.6

            # Check whether step has become too small
            if norm(t * d, 1) < optTol or t == 0:
                if verbose == 3:
                    print 'Line Search failed'
                t = 0
                f_new = f
                g_new = g
                break

            # Evaluate New Point
            x_new = x + t * d
            f_new, g_new = funObj(x_new)
            funEvals += 1

        # Take Step
        x = x_new
        f = f_new
        g = g_new

        optCond = sum(abs(funProj(x-g)-x))
        optCond = norm(funProj(x - g) - x, 1)
        projects += 1

        # Output Log
        if verbose >= 2:
            print '%10d %10d %10d %15.5e %15.5e %15.5e' % (
                    i, funEvals * funEvalMultiplier, projects, t, f, optCond)

        # Check optimality
        if optCond < optTol:
            print 'First-Order Optimality Conditions Below optTol'
            break

        if norm(t * d, 1) < optTol:
            if verbose >= 1:
                print 'Step size below optTol'
            break

        if np.fabs(f - f_old) < optTol:
            if verbose >= 1:
                print 'Function value changing by less than optTol'
            break

        if funEvals * funEvalMultiplier > maxIter:
            if verbose >= 1:
                print 'Function Evaluations exceeds maxIter'
            break

        if projects > maxProject:
            if verbose >= 1:
                print 'Number of projections exceeds maxProject'
            break

        i += 1

    return x, f, funEvals


def solveSubProblem(x, g, HvFunc, funProj, optTol, maxIter, testOpt,
                    feasibleInit, x_init):
    # Uses SPG to solve for projected quasi-Newton direction

    def subHv(p):
        d = p - x
        Hd = HvFunc(d)
        return g.dot(d) + d.dot(Hd) / 2, g + Hd

    p, f, funEvals, subProjects = minConF_SPG(subHv, x_init, funProj,
                                              verbose=0,
                                              optTol=optTol,
                                              maxIter=maxIter,
                                              testOpt=testOpt,
                                              feasibleInit=feasibleInit)
    return p, subProjects

