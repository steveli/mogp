from __future__ import division
import numpy as np
from numpy.linalg import solve
from scipy import optimize
#from scipy.optimize import approx_fprime
#from checkgrad import checkgrad


MEAN_SHIFT = True

def cov_mat(x1, x2, a, b):
    return a * np.exp(-b * (x1[:, np.newaxis] - x2)**2)


def reg_cov_mat(x, a, b, c):
    return cov_mat(x, x, a, b) + c * np.eye(x.shape[0])


def posterior_mean_cov(t_train, y_train, t_test, parms, mean_shift=MEAN_SHIFT):
    """Prediction for Gaussian process regression

    Returns the predictive mean and covariance matrix

    Parameters
    ----------
    t_train : array_like, shape (n_training_samples, )
              time points of training samples

    y_train : array_like, shape (n_training_samples, )
              values of corresponding time points of training samples

    t_test : array_like, shape (n_test_samples, )
             time points of test samples

    parms : tuple, length 3
            hyperparameters (a, b, c) for covariance function of Gaussian
            process.
            [K(t)]_ij = a * exp(-b * (t_i - t_j)^2) + c * I(i == j)

    Returns
    -------
    mean_test : array, shape (n_test_samples, )
                predictive mean

    cov_test : array, shape (n_test_samples, n_test_samples)
               predictive covariance matrix
    """

    #assert t_train.shape[0] == y_train.shape[0]
    a, b, c = parms

    if len(y_train) == 0:
        return np.zeros_like(t_test), cov_mat(t_test, t_test, a, b)

    K_train = reg_cov_mat(t_train, a, b, c)
    K_train_test = cov_mat(t_train, t_test, a, b)

    Ktr_inv_Ktt = solve(K_train, K_train_test)

    if mean_shift:
        mu = np.mean(y_train)
        mean_test = mu + Ktr_inv_Ktt.T.dot(y_train - mu)
    else:
        mean_test = Ktr_inv_Ktt.T.dot(y_train)

    cov_test = cov_mat(t_test, t_test, a, b) - K_train_test.T.dot(Ktr_inv_Ktt)

    return mean_test, cov_test


def pointwise_posterior_mean_var(t_train, y_train, t_test, parms,
                                 mean_shift=MEAN_SHIFT):
    """Prediction for Gaussian process regression

    Returns the pointwise predictive mean and variance

    Parameters
    ----------
    t_train : array_like, shape (n_training_samples, )
              time points of training samples

    y_train : array_like, shape (n_training_samples, )
              values of corresponding time points of training samples

    t_test : array_like, shape (n_test_samples, )
             time points of test samples

    parms : tuple, length 3
            hyperparameters (a, b, c) for covariance function of Gaussian
            process.
            [K(t)]_ij = a * exp(-b * (t_i - t_j)^2) + c * I(i == j)

    Returns
    -------
    mean_test : array, shape (n_test_samples, )
                predictive mean

    var_test : array, shape (n_test_samples, )
               predictive variance
    """

    #assert t_train.shape[0] == y_train.shape[0]
    a, b, c = parms

    if len(y_train) == 0:
        return np.zeros_like(t_test), np.ones_like(t_test) * a

    K_train = reg_cov_mat(t_train, a, b, c)
    K_train_test = cov_mat(t_train, t_test, a, b)

    Ktr_inv_Ktt = solve(K_train, K_train_test)

    if mean_shift:
        mu = np.mean(y_train)
        mean_test = mu + Ktr_inv_Ktt.T.dot(y_train - mu)
    else:
        mean_test = Ktr_inv_Ktt.T.dot(y_train)

    var_test = a * np.ones_like(t_test) - (K_train_test * Ktr_inv_Ktt).sum(axis=0)

    return mean_test, var_test


def posterior_mean(t_train, y_train, t_test, parms, mean_shift=MEAN_SHIFT):
    a, b, c = parms

    K_train = reg_cov_mat(t_train, a, b, c)
    K_test_train = cov_mat(t_test, t_train, a, b)

    if mean_shift:
        mu = np.mean(y_train)
        post_mean = mu + K_test_train.dot(solve(K_train, y_train - mu))
    else:
        post_mean = K_test_train.dot(solve(K_train, y_train))

    return post_mean


def _dfunc(dk, cov, Kinv_y):
    return (Kinv_y.dot(dk).dot(Kinv_y) - np.trace(solve(cov, dk))) * 0.5


def learn_hyperparms(ts,
                     a_shape=None, a_mean=None,
                     b_shape=None, b_mean=None,
                     c_shape=None, c_mean=None,
                     mean_shift=MEAN_SHIFT):

    if a_shape is not None:
        a_scale = a_mean / a_shape

    if b_shape is not None:
        b_scale = b_mean / b_shape

    if c_shape is not None:
        c_scale = c_mean / c_shape

    #@checkgrad
    def neg_mloglik(w):
        #print 'parm', w
        a, b, c = w
        f = 0
        df = np.zeros_like(w)

        for t, y_orig in ts:
            if mean_shift:
                y = y_orig - np.mean(y_orig)
            else:
                y = y_orig
            n = t.shape[0]
            tsq = (t[:, np.newaxis] - t)**2
            da = np.exp(-b * tsq)
            cov = a * da
            db = -tsq * cov
            cov += c * np.eye(n)
            dc = np.eye(n)
            log_det = np.linalg.slogdet(cov)[1]
            Kinv_y = solve(cov, y)
            f += -0.5 * (y.dot(Kinv_y) + log_det)

            df[0] += _dfunc(da, cov, Kinv_y)
            df[1] += _dfunc(db, cov, Kinv_y)
            df[2] += _dfunc(dc, cov, Kinv_y)

        if a_shape is not None:
            f += (a_shape - 1) * np.log(a) - a / a_scale
            df[0] += (a_shape - 1) / a - 1 / a_scale

        if b_shape is not None:
            f += (b_shape - 1) * np.log(b) - b / b_scale
            df[1] += (b_shape - 1) / b - 1 / b_scale

        if c_shape is not None:
            f += (c_shape - 1) * np.log(c) - c / c_scale
            df[2] += (c_shape - 1) / c - 1 / c_scale

        #print -f, -df

        return -f, -df

    w0 = np.array([1, 200, 1e-1])
    opt_parms = optimize.fmin_l_bfgs_b(neg_mloglik, w0,
            #bounds=[(1e-3, None), (1e-3, None), (1e-3, None)],
            bounds=[(1e-3, None), (1e-5, None), (1e-5, None)],
            factr=1e3, pgtol=1e-07, #disp=1,
            )[0]

    #print neg_mloglik(opt_parms)[1]
    #def _func(x, *args): return neg_mloglik(x, *args)[0]
    #print approx_fprime(opt_parms, _func, 1e-8)

    print opt_parms
    return opt_parms
