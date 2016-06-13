def SquaredError(w, X, y):
    # w(feature,1)
    # X(instance,feature)
    # y(instance,1)

    # Explicitly form X'X and do 2 matrix-vector product
    n, p = X.shape
    XX = X.T.dot(X)

    if n < p:    # Do two matrix-vector products with X
        Xw = X.dot(w)
        res = Xw - y
        f = (res**2).sum()
        g = 2 * (X.T.dot(res))
    else:    # Do 1 matrix-vector product with X and 1 with X'X
        XXw = XX.dot(w)
        Xy = X.T.dot(y)
        f = w.dot(XXw) - 2 * w.dot(Xy) + y.dot(y)
        g = 2 * XXw - 2 * Xy

    #H = 2 * XX
    #return f, g, H
    return f, g
