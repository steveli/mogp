from __future__ import division
import numpy as np
from scipy.linalg import solve


def polyinterp(points):
    """Minimum of interpolating polynomial based on function and derivative
    values

    In can also be used for extrapolation if {xmin,xmax} are outside
    the domain of the points.

    Input:
        points(pointNum,[x f g])
        xmin: min value that brackets minimum (default: min of points)
        xmax: max value that brackets maximum (default: max of points)

    set f or g to sqrt(-1) if they are not known
    the order of the polynomial is the number of known f and g values minus 1
    """

    nPoints = points.shape[0]
    order = (np.isreal(points[:, 1:3])).sum() - 1

    # Code for most common case:
    #   - cubic interpolation of 2 points
    #       w/ function and derivative values for both
    #   - no xminBound/xmaxBound

    if nPoints == 2 and order == 3:
        # Solution in this case (where x2 is the farthest point):
        #    d1 = g1 + g2 - 3*(f1-f2)/(x1-x2)
        #    d2 = sqrt(d1^2 - g1*g2)
        #    minPos = x2 - (x2 - x1)*((g2 + d2 - d1)/(g2 - g1 + 2*d2))
        #    t_new = min(max(minPos,x1),x2)
        if points[0, 1] < points[1, 1]:
            x_lo, x_hi = points[0, 0], points[1, 0]
            f_lo, f_hi = points[0, 1], points[1, 1]
            g_lo, g_hi = points[0, 2], points[1, 2]
        else:
            x_lo, x_hi = points[1, 0], points[0, 0]
            f_lo, f_hi = points[1, 1], points[0, 1]
            g_lo, g_hi = points[1, 2], points[0, 2]
        d1 = g_lo + g_hi - 3 * (f_lo - f_hi) / (x_lo - x_hi)
        d2 = np.sqrt(d1 * d1 - g_lo * g_hi)
        if np.isreal(d2):
            t = x_hi - (x_hi - x_lo) * ((g_hi + d2 - d1) /
                                        (g_hi - g_lo + 2 * d2))
            minPos = min(max(t, x_lo), x_hi)
        else:
            minPos = (x_lo + x_hi) / 2
        return minPos

    xmin = min(points[:, 0])
    xmax = max(points[:, 0])

    # Compute Bounds of Interpolation Area

    xminBound = xmin
    xmaxBound = xmax

    # Constraints Based on available Function Values
    A = np.zeros((0, order + 1))
    b = []

    for i in xrange(nPoints):
        if np.isreal(points[i, 1]):
            constraint = np.zeros(order + 1)
            for j in xrange(order + 1):
                constraint[order - j] = points[i, 0]**j
            A = np.vstack((A, constraint))
            b = np.append(b, points[i, 1])

    # Constraints based on available Derivatives
    for i in xrange(nPoints):
        if np.isreal(points[i, 2]):
            constraint = np.zeros(order + 1)
            for j in xrange(order):
                constraint[j] = (order - j) * points[i, 0]**(order - j - 1)
            A = np.vstack((A, constraint))
            b = np.append(b, points[i, 2])

    # Find interpolating polynomial
    params = solve(A, b)

    # Compute Critical Points
    dParams = np.zeros(order)
    for i in xrange(len(params) - 1):
        dParams[i] = params[i] * (order - i)

    if np.any(np.isinf(dParams)):
        cp = np.concatenate((np.array([xminBound, xmaxBound]),
                             points[:, 0]))
    else:
        cp = np.concatenate((np.array([xminBound, xmaxBound]),
                             points[:, 0]),
                             np.roots(dParams))

    # Test Critical Points
    fmin = np.inf
    # Default to Bisection if no critical points valid
    minPos = (xminBound + xmaxBound) / 2
    for xCP in cp:
        if np.isreal(xCP) and xCP >= xminBound and xCP <= xmaxBound:
            fCP = np.polyval(params, xCP)
            if np.isreal(fCP) and fCP < fmin:
                minPos = np.real(xCP)
                fmin = np.real(fCP)

    return minPos
