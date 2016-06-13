from __future__ import division
import numpy as np
from numpy.linalg import norm
from isLegal import isLegal
from polyinterp import polyinterp


def minConF_SPG(funObj, x, funProj,
                verbose=2,
                numDiff=0,
                optTol=1e-6,
                maxIter=500,
                suffDec=1e-4,
                interp=2,
                memory=10,
                useSpectral=1,
                curvilinear=0,
                feasibleInit=0,
                testOpt=1,
                bbType=1):

    """Function for using Spectral Projected Gradient to solve problems
    of the form:

    min funObj(x) s.t. x in C

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
        numDiff:
            compute derivatives numerically
            (0: use user-supplied derivatives (default),
             1: use finite differences,
             2: use complex differentials)
        suffDec:
            sufficient decrease parameter in Armijo condition (default: 1e-4)
        interp:
            type of interpolation
            (0: step-size halving, 1: quadratic, 2: cubic)
        memory:
            number of steps to look back in non-monotone Armijo condition
        useSpectral:
            use spectral scaling of gradient direction (default: 1)
        curvilinear:
            backtrack along projection Arc (default: 0)
        testOpt:
            test optimality condition (default: 1)
        feasibleInit:
            if 1, then the initial point is assumed to be feasible
        bbType:
            type of Barzilai Borwein step (default: 1)

    Notes:
        - if the projection is expensive to compute, you can reduce the
          number of projections by setting testOpt to 0
    """

    #nVars = len(x)

    # Output Log
    if verbose >= 2:
        if testOpt:
            print '%10s %10s %10s %15s %15s %15s' % (
                    'Iteration', 'FunEvals', 'Projections',
                    'Step Length', 'Function Val', 'Opt Cond')
        else:
            print '%10s %10s %10s %15s %15s' % (
                    'Iteration', 'FunEvals', 'Projections',
                    'Step Length', 'Function Val')

    # Make objective function (if using numerical derivatives)
    funEvalMultiplier = 1
    # FIXME
    #if numDiff:
    #    if numDiff == 2:
    #        useComplex = 1
    #    else:
    #        useComplex = 0
    #    funObj = @(x)autoGrad(x,useComplex,funObj)
    #    funEvalMultiplier = nVars+1-useComplex
    #end

    # Evaluate Initial Point
    if not feasibleInit:
        x = funProj(x)

    f, g = funObj(x)
    projects = 1
    funEvals = 1

    # Optionally check optimality
    if testOpt:
        projects += 1
        if norm(funProj(x - g) - x, 1) < optTol:
            if verbose >= 1:
                print 'First-Order Optimality Conditions Below optTol at Initial Point'
            return x, f, funEvals, projects

    i = 0
    f_prev, t_prev = 0, 0

    while funEvals <= maxIter:

        # Compute Step Direction
        if i == 0 or not useSpectral:
            alpha = 1
        else:
            y = g - g_old
            s = x - x_old
            if bbType == 1:
                alpha = s.dot(s) / s.dot(y)
            else:
                alpha = s.dot(y) / y.dot(y)

            if alpha <= 1e-10 or alpha > 1e10:
                alpha = 1

        d = -alpha * g
        f_old = f
        x_old = x
        g_old = g

        # Compute Projected Step
        if not curvilinear:
            d = funProj(x + d) - x
            projects += 1

        # Check that Progress can be made along the direction
        gtd = g.dot(d)
        if gtd > -optTol:
            if verbose >= 1:
                print 'Directional Derivative below optTol'
            break

        # Select Initial Guess to step length
        if i == 0:
            t = min(1, 1 / norm(g, 1))
        else:
            t = 1

        # Compute reference function for non-monotone condition

        if memory == 1:
            funRef = f
        else:
            if i == 0:
                old_fvals = np.empty(memory)
                old_fvals.fill(-np.inf)

            if i < memory:
                old_fvals[i] = f
            else:
                old_fvals = np.concatenate((old_fvals[1:], f))

            funRef = old_fvals.max()

        # Evaluate the Objective and Gradient at the Initial Step
        if curvilinear:
            x_new = funProj(x + t * d)
            projects += 1
        else:
            x_new = x + t * d

        f_new, g_new = funObj(x_new)
        funEvals += 1

        # Backtracking Line Search
        lineSearchIters = 1
        while f_new > funRef + suffDec * g.dot(x_new-x) or not isLegal(f_new):
            temp = t
            if interp == 0 or not isLegal(f_new):
                if verbose == 3:
                    print 'Halving Step Size'
                t = t / 2
            elif interp == 2 and isLegal(g_new):
                if verbose == 3:
                    print 'Cubic Backtracking'
                t = polyinterp(np.array([[0, f, gtd],
                                         [t, f_new, g_new.dot(d)]]))
            elif lineSearchIters < 2 or not isLegal(f_prev):
                if verbose == 3:
                    print 'Quadratic Backtracking'
                t = polyinterp(np.array([[0, f, gtd],
                                         [t, f_new, np.sqrt(-1)]]))
            else:
                if verbose == 3:
                    print 'Cubic Backtracking on Function Values'
                t = polyinterp(np.array([[0, f, gtd],
                                         [t, f_new, np.sqrt(-1)],
                                         [t_prev, f_prev, np.sqrt(-1)]]))

            # Adjust if change is too small
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
            f_prev = f_new
            t_prev = temp
            if curvilinear:
                x_new = funProj(x + t * d)
                projects += 1
            else:
                x_new = x + t * d
            f_new, g_new = funObj(x_new)
            funEvals += 1
            lineSearchIters += 1

        # Take Step
        x = x_new
        f = f_new
        g = g_new

        if testOpt:
            optCond = norm(funProj(x - g) - x, 1)
            projects += 1

        # Output Log
        if verbose >= 2:
            if testOpt:
                print '%10d %10d %10d %15.5e %15.5e %15.5e' % (
                        i, funEvals * funEvalMultiplier, projects,
                        t, f, optCond)
            else:
                print '%10d %10d %10d %15.5e %15.5e' % (
                        i, funEvals * funEvalMultiplier, projects, t, f)

        # Check optimality
        if testOpt:
            if optCond < optTol:
                if verbose >= 1:
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

        i += 1

    return x, f, funEvals, projects

