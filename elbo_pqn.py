from __future__ import division
import time
import numpy as np
from numpy.linalg import solve
from numpy.linalg import inv
from scipy.optimize import fmin_l_bfgs_b
from scipy.stats import norm
import pylab as pl
import argparse
import gp
from pickle_io import pickle_load, pickle_save

from pyPQN.groupl1_makeGroupPointers import groupl1_makeGroupPointers
from pyPQN.auxGroupL2Project import projectAux
from pyPQN.minConF_PQN import minConf_PQN
from pyPQN.projectRandom2 import randomProject


_epsilon = np.sqrt(np.finfo(float).eps)

gp_noise = 1e-2

regularize_h = False


def update_learning_rate(lmbda, prev_lrate):
    return 1 / (lmbda + 1 / prev_lrate)


class SharedParameters(object):
    def __init__(self, z, P, Q):
        self.z = z
        self.w = np.ones((P, Q))
        #self.w = np.zeros((P, Q))   # w won't update, so do mg's
        self.beta = 1 / 0.5 * np.ones(P)
        #self.beta = 1 / np.var(np.concatenate(y)) * np.ones(P)

        self.beta_lbound = 1e-5
        #self.ag_lbound = 0.1
        #self.bg_lbound = .8
        #self.ah_lbound = 0.1
        #self.bh_lbound = .8
        #self.ag_lbound = 1
        self.ag_lbound = 1e-2
        #self.bg_lbound = 200
        self.bg_lbound = 1e-2
        #self.ah_lbound = .1
        self.ah_lbound = 1e-2
        #self.bh_lbound = 200
        self.bh_lbound = 1e-2

        self.g_parms = np.empty((Q, 2))
        self.g_parms[:, 0] = 1
        self.g_parms[:, 1] = 100
        self.h_parms = np.empty((P, 2))
        self.h_parms[:, 0] = .1
        self.h_parms[:, 1] = 100

        self.g_gp_b = np.empty(Q)
        self.h_gp_a = np.empty(P)
        self.h_gp_b = np.empty(P)

        self.g_gp_b.fill(100)
        self.h_gp_a.fill(.1)
        self.h_gp_b.fill(100)

        self.reg_ah = 0
        #self.reg_ah = 5
        #self.reg_ah = 10
        #self.reg_ah = 1
        #self.reg_ag = 10
        self.reg_ag = 0

        self.delta_w = np.zeros_like(self.w)
        self.momentum_w = 0.9
        #self.lrate_w = 1e-5
        self.lrate_w = 1e-3
        self.delta_beta = np.zeros_like(self.beta)
        self.momentum_beta = 0.9
        #self.lrate_beta = 1e-4
        self.lrate_beta = 1e-3
        self.delta_g_parms = np.zeros_like(self.g_parms)
        self.momentum_g_parms = 0.9
        #self.lrate_g_parms = 1e-5
        self.lrate_g_parms = 1e-3
        self.delta_h_parms = np.zeros_like(self.h_parms)
        self.momentum_h_parms = 0.9
        #self.lrate_h_parms = 1e-5
        self.lrate_h_parms = 1e-3

        #self.lrate_w = 1e-6
        #self.lrate_beta = self.lrate_g_parms = self.lrate_h_parms = 1e-5

        self.lrate_w = self.lrate_beta = self.lrate_g_parms \
                     = self.lrate_h_parms = 1e-4

        self.momentum_w = self.momentum_beta = self.momentum_g_parms \
                        = self.momentum_h_parms = 0.7

        self.P = P
        self.Q = Q

    def update_lrate(self):
        lmbda = 400
        self.lrate_w = update_learning_rate(lmbda, self.lrate_w)
        self.lrate_beta = update_learning_rate(lmbda, self.lrate_beta)
        self.lrate_g_parms = update_learning_rate(lmbda, self.lrate_g_parms)
        self.lrate_h_parms = update_learning_rate(lmbda, self.lrate_h_parms)
        print '-' * 20
        print self.lrate_w


def logdet(cov):
    return np.linalg.slogdet(cov)[1]


def cov_mat(x1, x2, a, b):
    return a * np.exp(-b * (x1[:, np.newaxis] - x2)**2)


def reg_cov_mat(x, a, b, c):
    return cov_mat(x, x, a, b) + c * np.eye(x.shape[0])


def compute_dK(x, y, a, b):
    diff_sq = -(x[:, np.newaxis] - y)**2
    dK_da = np.exp(b * diff_sq)
    K = a * dK_da
    dK_db = K * diff_sq
    return K, dK_da, dK_db


def diag_posterior_cov(x, z, a, b, noise=0):
    K_train = reg_cov_mat(z, a, b, noise)
    K_train_test = cov_mat(z, x, a, b)
    Ktr_inv_Ktt = solve(K_train, K_train_test)
    return a * np.ones_like(x) - (K_train_test * Ktr_inv_Ktt).sum(axis=0)


def update(w, delta_w, grad_w, momentum, lrate):
    delta_w = momentum * delta_w + lrate * grad_w
    return w + delta_w, delta_w


def gp_predict_approx(a, b, m, S, z, x_test, regularize=0):
    L = len(x_test)
    M = len(z)
    Ktz = cov_mat(x_test, z, a, b)
    Kzz = cov_mat(z, z, a, b) + regularize * np.eye(M)
    Kzz_inv = inv(Kzz)
    mu = Ktz.dot(solve(Kzz, m))
    var = a * np.ones(L) - (Ktz.dot(Kzz_inv.dot(np.eye(M) - S.dot(Kzz_inv)))
                            * Ktz).sum(axis=1)
    #cov = (cov_mat(x_test, x_test, a, b)
    #        - (Ktz.dot(Kzz_inv.dot(np.eye(M) - S.dot(Kzz_inv))).dot(Ktz.T)))
    Kzzinv_Ktz = solve(Kzz, Ktz.T)
    cov = (cov_mat(x_test, x_test, a, b)
            - Kzzinv_Ktz.T.dot(Ktz.T)
            + Kzzinv_Ktz.T.dot(S).dot(Kzzinv_Ktz))
    return mu, var, cov


class TimeSeries(object):
    def __init__(self, x, y, shared_parms):
        self.x_raw = x
        self.y_raw = y
        self.x = [np.asarray(x_) for x_ in x if len(x_) > 0]
        self.y = [np.asarray(y_) for y_ in y if len(y_) > 0]
        self.z = z = shared_parms.z    # inducing points
        self.N = np.array([len(x_) for x_ in self.x])
        self.M = M = len(z)
        self.shared_parms = shared_parms
        self.P = P = shared_parms.P
        self.Q = Q = shared_parms.Q
        self.idx_nz = idx_nz = [i for i in xrange(P) if len(x[i]) > 0]
        self.inv_idx = {j: i for (i, j) in enumerate(self.idx_nz)}
        self.P_nz = P_nz = len(idx_nz)
        self.mh = np.zeros((P_nz, M))
        self.Sh = np.zeros((P_nz, M, M))
        for i in xrange(P_nz):
            subset_len = int(len(self.y[i]) * .6)
            if subset_len > 1:
                y_subset = np.random.choice(self.y[i], subset_len)
                # min(x, 100) in case that y_subset contains
                # identical elements.
                var = np.var(y_subset)
                var_inv = 1 / var if var > 0 else 100
            else:
                var_inv = 10
            var_inv = 10
            np.fill_diagonal(self.Sh[i], var_inv)
            #np.fill_diagonal(self.Sh[i], 1)
        self.mg = np.zeros((Q, M))
        self.Sg = np.zeros((Q, M, M))

        y_cat = np.concatenate(self.y)
        y_cat_subset_size = int(len(y_cat) * .6)
        for j in xrange(Q):
            if y_cat_subset_size > 1:
                y_cat_subset = np.random.choice(y_cat, y_cat_subset_size)
                var = np.var(y_cat_subset)
                var_inv = 1 / var if var > 0 else 100
            else:
                var_inv = 10
            np.fill_diagonal(self.Sg[j], var_inv)
            #np.fill_diagonal(self.Sg[j], 1)
        #self.Sg, self.mg = pickle_load('ms.pkl')

    def compute_A(self, idx, a, b, noise=0):
        z = self.z
        Kmm = cov_mat(z, z, a, b) + noise * np.eye(self.M)
        Kmn = cov_mat(z, self.x[idx], a, b)
        A = solve(Kmm, Kmn).T    # N_i by M
        return A

    def compute_Kinv(self, kernel_parms, noise=0):
        a, b = kernel_parms
        z = self.z
        Kmm = cov_mat(z, z, a, b) + noise * np.eye(self.M)
        return inv(Kmm)

    def compute_K(self, a, b, noise=0):
        z = self.z
        Kmm = cov_mat(z, z, a, b) + noise * np.eye(self.M)
        return Kmm, inv(Kmm)

    def compute_elbo(self, var=None):
        P, Q = self.P, self.Q
        P_nz = self.P_nz
        M = self.M
        N = self.N
        idx_nz = self.idx_nz
        x = self.x
        y = self.y
        mg = self.mg
        Sg = self.Sg
        mh = self.mh
        z = self.shared_parms.z

        if var is None:
            w = self.shared_parms.w[idx_nz]
            beta = self.shared_parms.beta[idx_nz]
            g_gp_b = self.shared_parms.g_gp_b
            h_gp_a = self.shared_parms.h_gp_a[idx_nz]
            h_gp_b = self.shared_parms.h_gp_b[idx_nz]
        else:
            w = var[:(P * Q)].reshape((P, Q))[idx_nz]
            var = var[(P * Q):]
            beta = var[:P][idx_nz]
            var = var[P:]
            g_gp_b = var[:Q]
            var = var[Q:]
            h_gp_a = var[:P][idx_nz]
            var = var[P:]
            h_gp_b = var[:P][idx_nz]

        #noise_parm = 1e-10
        #noise_parm = 0
        noise_parm = gp_noise

        # update variational parameters of g_j: m_j, S_j

        Am = np.empty((P_nz, Q), dtype=object)   # each entry: (N_i,)
        wAm = np.empty((P_nz, Q), dtype=object)  # each entry: (N_i,)
        sum_wAm = np.empty(P_nz, dtype=object)   # each entry: (N_i,)
        Ag = np.empty((P_nz, Q), dtype=object)   # each entry: (N_i, M)
        Ah = np.empty(P_nz, dtype=object)        # each entry: (N_i, M)
        # y0_i = y_i - A_i m_i - sum_j w_ij A_ij m_j
        y0 = np.empty(P_nz, dtype=object)        # each entry: (N_i,)
        for i in xrange(P_nz):
            wAm_ = np.empty((Q, self.N[i]))
            for j in xrange(Q):
                A = self.compute_A(i, 1, g_gp_b[j], noise_parm)
                Ag[i, j] = A
                Am[i, j] = Am_ = A.dot(mg[j])
                wAm_[j] = wAm[i, j] = w[i, j] * Am_
            sum_wAm[i] = wAm_.sum(axis=0)
            Ah[i] = Ai = self.compute_A(i, h_gp_a[i], h_gp_b[i], noise_parm)
            y0[i] = y[i] - Ai.dot(mh[i]) - sum_wAm[i]

        Kg = np.empty(Q, dtype=object)
        Kg_inv = np.empty(Q, dtype=object)
        Kh = np.empty(P_nz, dtype=object)
        Kh_inv = np.empty(P_nz, dtype=object)

        for i in xrange(P_nz):
            Kh[i], Kh_inv[i] = self.compute_K(h_gp_a[i], h_gp_b[i], noise_parm)

        for j in xrange(Q):
            Kg[j], Kg_inv[j] = self.compute_K(1, g_gp_b[j], noise_parm)

        # compute evidence lower bound
        Sg = self.Sg
        mg = self.mg
        Sh = self.Sh

        tKg = np.empty((P_nz, Q))         # sum_n [tilde{K_ij}]_nn
        tKh = np.empty(P_nz)              # sum_n [tilde{K_i}]_nn
        sum_ASAg = np.empty((P_nz, Q))    # sum_n A_ijn S_j A_ijn'
        sum_ASAh = np.empty(P_nz)         # sum_n A_in S_i A_in'

        for i in xrange(P_nz):
            tKh[i] = diag_posterior_cov(x[i], self.z,
                                        h_gp_a[i], h_gp_b[i],
                                        noise_parm).sum()
            sum_ASAh[i] = (Ah[i].dot(Sh[i]) * Ah[i]).sum()
            for j in xrange(Q):
                tKg[i, j] = diag_posterior_cov(x[i], self.z,
                                               1, g_gp_b[j],
                                               noise_parm).sum()
                sum_ASAg[i, j] = (Ag[i, j].dot(Sg[j]) * Ag[i, j]).sum()

        w2 = w**2
        # sum_j w_ij^2 sum_n (A_ijn S_j A_ijn' + [tilde{K_ij}]_nn)
        sum_2g = (w2 * (tKg + sum_ASAg)).sum(axis=1)   # shape: (P,)
        elbo = -0.5 * ((sum_2g + tKh + sum_ASAh) * beta).sum()

        for i in xrange(P_nz):
            wAm_ = np.empty((Q, self.N[i]))
            for j in xrange(Q):
                Am[i, j] = Am_ = Ag[i, j].dot(mg[j])
                wAm_[j] = wAm[i, j] = w[i, j] * Am[i, j]
            y0[i] = y[i] - Ah[i].dot(mh[i]) - wAm_.sum(axis=0)

        y0sq = np.empty(P_nz)
        for i in xrange(P_nz):
            y0sq[i] = y0[i].dot(y0[i])
            elbo += 0.5 * (np.log(beta[i]) * N[i]
                           - beta[i] * y0sq[i]
                           + logdet(Sh[i])
                           - logdet(Kh[i])
                           - (Kh_inv[i].T * Sh[i]).sum()
                           - mh[i].dot(Kh_inv[i]).dot(mh[i]))

        for j in xrange(Q):
            elbo += 0.5 * (logdet(Sg[j])
                           - logdet(Kg[j])
                           - (Kg_inv[j].T * Sg[j]).sum()
                           - mg[j].dot(Kg_inv[j]).dot(mg[j]))


        if regularize_h:
            elbo -= self.shared_parms.reg_ah * h_gp_a.dot(h_gp_a)

        return -elbo

    def update_m_S(self):
        P, Q = self.P, self.Q
        P_nz = self.P_nz
        M = self.M
        N = self.N
        idx_nz = self.idx_nz
        x = self.x
        y = self.y
        z = self.z
        mg = self.mg
        Sg = self.Sg
        mh = self.mh
        w = self.shared_parms.w[idx_nz]
        beta = self.shared_parms.beta[idx_nz]

        g_gp_b = self.shared_parms.g_gp_b
        h_gp_a = self.shared_parms.h_gp_a[idx_nz]
        h_gp_b = self.shared_parms.h_gp_b[idx_nz]

        #noise_parm = 1e-10
        #noise_parm = 0
        noise_parm = gp_noise

        # update variational parameters of g_j: m_j, S_j

        Ag = np.empty((P_nz, Q), dtype=object)   # each entry: (N_i, M)
        Ah = np.empty(P_nz, dtype=object)        # each entry: (N_i, M)
        # y0_i = y_i - A_i m_i - sum_j w_ij A_ij m_j

        for i in xrange(P_nz):
            Ah[i] = self.compute_A(i, h_gp_a[i], h_gp_b[i], noise_parm)
            for j in xrange(Q):
                Ag[i, j] = self.compute_A(i, 1, g_gp_b[j], noise_parm)

        QM = Q * M
        PM = P_nz * M
        stacked_len = QM + PM
        coef = np.zeros((stacked_len, stacked_len))
        cat_y = np.zeros(stacked_len)
        for j in xrange(Q):
            idx_lo = j * M
            idx_hi = idx_lo + M

            Kg_inv = self.compute_K(1, g_gp_b[j], noise_parm)[1]
            coef[idx_lo:idx_hi, idx_lo:idx_hi] = Kg_inv

            for i in xrange(P_nz):
                cat_y[idx_lo:idx_hi] += (
                        beta[i] * w[i, j] * Ag[i, j].T.dot(y[i]))
                h_lo = QM + i * M
                h_hi = h_lo + M
                coef[idx_lo:idx_hi, h_lo:h_hi] = (beta[i] * w[i, j] *
                                                  Ag[i, j].T.dot(Ah[i]))
                for k in xrange(Q):
                    coef[idx_lo:idx_hi, (k * M):(k * M + M)] += (
                            beta[i] * w[i, j] * w[i, k] *
                            Ag[i, j].T.dot(Ag[i, k]))

            self.Sg[j] = inv(coef[idx_lo:idx_hi, idx_lo:idx_hi])

        for i in xrange(P_nz):
            idx_lo = QM + M * i
            idx_hi = idx_lo + M

            cat_y[idx_lo:idx_hi] = beta[i] * Ah[i].T.dot(y[i])
            Kh_inv = self.compute_K(h_gp_a[i], h_gp_b[i], noise_parm)[1]
            coef[idx_lo:idx_hi, idx_lo:idx_hi] = inv_Sh = (
                    Kh_inv + beta[i] * Ah[i].T.dot(Ah[i]))
            for j in xrange(Q):
                g_lo = j * M
                g_hi = g_lo + M
                coef[idx_lo:idx_hi, g_lo:g_hi] = (beta[i] * w[i, j] *
                                                  Ah[i].T.dot(Ag[i, j]))
            self.Sh[i] = inv(inv_Sh)

        cat_m = np.linalg.solve(coef, cat_y)
        idx_lo = 0
        for j in xrange(Q):
            idx_hi = idx_lo + M
            self.mg[j] = cat_m[idx_lo:idx_hi]
            idx_lo = idx_hi
        for i in xrange(P_nz):
            idx_hi = idx_lo + M
            self.mh[i] = cat_m[idx_lo:idx_hi]
            idx_lo = idx_hi

    def slfm_learn(self, var, check_grad=False, check_elbo=False):
        P, Q = self.P, self.Q
        P_nz = self.P_nz
        M = self.M
        N = self.N
        idx_nz = self.idx_nz
        x = self.x
        y = self.y
        z = self.z
        mg = self.mg
        Sg = self.Sg
        mh = self.mh

        w = var[:(P * Q)].reshape((P, Q))[idx_nz]
        var = var[(P * Q):]
        beta = var[:P][idx_nz]
        var = var[P:]
        g_gp_b = var[:Q]
        var = var[Q:]
        h_gp_a = var[:P][idx_nz]
        var = var[P:]
        h_gp_b = var[:P][idx_nz]

        #noise_parm = 1e-10
        #noise_parm = 0
        noise_parm = gp_noise

        # update variational parameters of g_j: m_j, S_j

        Am = np.empty((P_nz, Q), dtype=object)   # each entry: (N_i,)
        wAm = np.empty((P_nz, Q), dtype=object)  # each entry: (N_i,)
        Ag = np.empty((P_nz, Q), dtype=object)   # each entry: (N_i, M)
        Ah = np.empty(P_nz, dtype=object)        # each entry: (N_i, M)
        # y0_i = y_i - A_i m_i - sum_j w_ij A_ij m_j
        y0 = np.empty(P_nz, dtype=object)        # each entry: (N_i,)
        yh = np.empty(P_nz, dtype=object)        # each entry: (N_i,)

        for i in xrange(P_nz):
            Ah[i] = Ai = self.compute_A(i, h_gp_a[i], h_gp_b[i], noise_parm)
            y0[i] = y[i] - Ai.dot(mh[i])
            yh[i] = np.copy(y0[i])
            for j in xrange(Q):
                A = self.compute_A(i, 1, g_gp_b[j], noise_parm)
                Ag[i, j] = A
                Am[i, j] = Am_ = A.dot(mg[j])
                wAm[i, j] = w[i, j] * Am_
                y0[i] -= wAm[i, j]
            if check_elbo:
                print '***', i
                print np.linalg.norm(y[i]), np.linalg.norm(Ai.dot(mh[i])), np.linalg.norm(mh[i])

        Kg = np.empty(Q, dtype=object)
        Kg_inv = np.empty(Q, dtype=object)
        Kh = np.empty(P_nz, dtype=object)
        Kh_inv = np.empty(P_nz, dtype=object)

        #print '-' * 10
        MQ = M * Q
        coef = np.zeros((MQ, MQ))
        cat_y = np.zeros(MQ)
        for j in xrange(Q):
            Kg[j], Kg_inv[j] = self.compute_K(1, g_gp_b[j], noise_parm)
            for i in xrange(P_nz):
                cat_y[(j * M):(j * M + M)] += (
                        beta[i] * w[i, j] * Ag[i, j].T.dot(yh[i]))
            for k in xrange(Q):
                wjk = w[i, j] * w[i, k]
                sub_coef = coef[(j * M):(j * M + M), (k * M):(k * M + M)]
                for i in xrange(P_nz):
                    sub_coef += beta[i] * wjk * Ag[i, j].T.dot(Ag[i, k])
                if k == j:
                    sub_coef += Kg_inv[j]

                    #tmp = np.linalg.eig(sub_coef)[0]
                    #print 'max, min', tmp.max(), tmp.min()

                    #self.Sg[j] = inv(sub_coef)

        #cat_mg = np.linalg.solve(coef, cat_y)
        #for j in xrange(Q):
        #    self.mg[j] = cat_mg[(j * M):(j * M + M)]
        #print np.linalg.norm(self.mg)

        #pickle_save('ms.pkl', self.Sg, self.mg)

        #print np.linalg.norm(self.mg)
        #raw_input()

        # compute evidence lower bound
        Sg = self.Sg
        mg = self.mg
        Sh = self.Sh

        tKg = np.empty((P_nz, Q))         # sum_n [tilde{K_ij}]_nn
        tKh = np.empty(P_nz)              # sum_n [tilde{K_i}]_nn
        sum_ASAg = np.empty((P_nz, Q))    # sum_n A_ijn S_j A_ijn'
        sum_ASAh = np.empty(P_nz)         # sum_n A_in S_i A_in'

        for i in xrange(P_nz):
            tKh[i] = diag_posterior_cov(x[i], self.z,
                                        h_gp_a[i], h_gp_b[i],
                                        noise_parm).sum()
            sum_ASAh[i] = (Ah[i].dot(Sh[i]) * Ah[i]).sum()
            for j in xrange(Q):
                tKg[i, j] = diag_posterior_cov(x[i], self.z,
                                               1, g_gp_b[j],
                                               noise_parm).sum()
                sum_ASAg[i, j] = (Ag[i, j].dot(Sg[j]) * Ag[i, j]).sum()

        w2 = w**2
        # sum_j w_ij^2 sum_n (A_ijn S_j A_ijn' + [tilde{K_ij}]_nn)
        sum_2g = (w2 * (tKg + sum_ASAg)).sum(axis=1)   # shape: (P_nz,)
        elbo = -0.5 * ((sum_2g + tKh + sum_ASAh) * beta).sum()
        if check_elbo:
            elbo1 = elbo
            print (sum_2g * beta).sum(), (tKh * beta).sum(), (sum_ASAh * beta).sum()
            print 'elbo 1', elbo1

        for i in xrange(P_nz):
            wAm_ = np.empty((Q, self.N[i]))
            for j in xrange(Q):
                Am[i, j] = Am_ = Ag[i, j].dot(mg[j])
                wAm_[j] = wAm[i, j] = w[i, j] * Am[i, j]
            y0[i] = y[i] - Ah[i].dot(mh[i]) - wAm_.sum(axis=0)

        for i in xrange(P_nz):
            Kh[i], Kh_inv[i] = self.compute_K(h_gp_a[i], h_gp_b[i], noise_parm)

        y0sq = np.empty(P_nz)
        #print beta
        for i in xrange(P_nz):
            y0sq[i] = y0[i].dot(y0[i])
            elbo += 0.5 * (np.log(beta[i]) * N[i]
                           - beta[i] * y0sq[i]
                           + logdet(Sh[i])
                           - logdet(Kh[i])
                           - (Kh_inv[i].T * Sh[i]).sum()
                           - mh[i].dot(Kh_inv[i]).dot(mh[i]))
            if check_elbo:
                print '###', i
                #print y0sq[i]
                print beta[i], y0sq[i], logdet(Sh[i]), logdet(Kh[i]), (Kh_inv[i].T * Sh[i]).sum(), mh[i].dot(Kh_inv[i]).dot(mh[i])

        if check_elbo:
            elbo2 = elbo
            print 'elbo 2', elbo2 - elbo1, elbo

        if check_grad:
            print '---'
            print np.log(beta[i]) * N[i]
            print beta[i] * y0sq[i]
            print logdet(Sh[i])
            print logdet(Kh[i])
            print (Kh_inv[i].T * Sh[i]).sum()
            print mh[i].dot(Kh_inv[i]).dot(mh[i])
            print np.linalg.eig(Kh[i])[0]
            print '---'
            print 'elbo-2', elbo

        for j in xrange(Q):
            elbo += 0.5 * (logdet(Sg[j])
                           - logdet(Kg[j])
                           - (Kg_inv[j].T * Sg[j]).sum()
                           - mg[j].dot(Kg_inv[j]).dot(mg[j]))

        if check_elbo:
            elbo3 = elbo
            print 'elbo 3', elbo3 - elbo2

        if regularize_h:
            elbo -= self.shared_parms.reg_ah * h_gp_a.dot(h_gp_a)

        if check_elbo:
            elbo4 = elbo
            print 'elbo 4', elbo4 - elbo3, elbo

        #print 'elbo', elbo

        ## FIXME
        #return elbo

        # update kernel hyperparameters of g_j
        # derivative of beta
        dbeta = 0.5 * (N / beta - y0sq - (sum_2g + tKh + sum_ASAh))

        # derivative of w
        dw = np.zeros((P_nz, Q))
        for i in xrange(P_nz):
            for j in xrange(Q):
                dw[i, j] = beta[i] * ((y0[i] * Am[i, j]).sum()
                                      - w[i, j] * (sum_ASAg[i, j] + tKg[i, j]))

        # derivative of kernel hyperparameters of g
        Knn = np.empty((P_nz, Q), dtype=object)
        Kmn = np.empty((P_nz, Q), dtype=object)
        Kmm = np.empty(Q, dtype=object)
        dKnn_da = np.empty((P_nz, Q), dtype=object)
        dKnn_db = np.empty((P_nz, Q), dtype=object)
        dKmn_da = np.empty((P_nz, Q), dtype=object)
        dKmn_db = np.empty((P_nz, Q), dtype=object)
        dKmm_da = np.empty(Q, dtype=object)
        dKmm_db = np.empty(Q, dtype=object)
        for j in xrange(Q):
            Kmm[j], dKmm_da[j], dKmm_db[j] = compute_dK(z, z, 1, g_gp_b[j])
            Kmm[j] += np.eye(M) * noise_parm
            for i in xrange(P_nz):
                Knn[i, j], dKnn_da[i, j], dKnn_db[i, j] = compute_dK(
                        x[i], x[i], 1, g_gp_b[j])
                Kmn[i, j], dKmn_da[i, j], dKmn_db[i, j] = compute_dK(
                        z, x[i], 1, g_gp_b[j])

        dL_db = np.zeros(Q)
        for j in xrange(Q):
            Kmm_inv = inv(Kmm[j])
            dL_db[j] = (-0.5 * np.trace(Kmm_inv.dot(dKmm_db[j]))
                        + 0.5 * np.trace(
                                Kmm_inv.dot(dKmm_db[j]).dot(Kmm_inv)
                                       .dot(Sg[j] + np.outer(mg[j], mg[j]))))

            for i in xrange(P_nz):
                A = Kmn[i, j].T.dot(Kmm_inv)
                dA_db = (dKmn_db[i, j].T - A.dot(dKmm_db[j])).dot(Kmm_inv)
                m_ = w[i, j] * mg[j]
                b = beta[i]

                bw2 = b * w[i, j]**2

                dL_db[j] += (b * y0[i].T.dot(dA_db).dot(m_)
                             - (0.5 * bw2 * np.trace(dKnn_db[i, j]
                                                     - A.dot(dKmn_db[i, j])
                                                     - dA_db.dot(Kmn[i, j])))
                             - bw2 * np.trace(A.dot(Sg[j]).dot(dA_db.T)))

        dw_ = np.zeros((P, Q))
        dw_[idx_nz] = dw
        dbeta_ = np.zeros(P)
        dbeta_[idx_nz] = dbeta

        comb_grad = np.concatenate((np.ravel(dw_), dbeta_, dL_db))

        return -comb_grad

    def update_m_S_h(self, check_grad=False):
        P, Q = self.P, self.Q
        P_nz = self.P_nz
        idx_nz = self.idx_nz
        M = self.M
        N = self.N
        x = self.x
        y = self.y
        z = self.z
        mg = self.mg
        Sh = self.Sh
        mh = self.mh
        w = self.shared_parms.w[idx_nz]
        beta = self.shared_parms.beta[idx_nz]

        g_gp_b = self.shared_parms.g_gp_b
        h_gp_a = self.shared_parms.h_gp_a[idx_nz]
        h_gp_b = self.shared_parms.h_gp_b[idx_nz]

        #noise_parm = 1e-10
        #noise_parm = 0
        noise_parm = gp_noise

        #Am = np.empty(P, dtype=object)   # each entry: (N_i,)
        Ah = np.empty(P_nz, dtype=object)        # each entry: (N_i, M)
        # y0_i = y_i - A_i m_i - sum_j w_ij A_ij m_j
        y0 = np.empty(P_nz, dtype=object)        # each entry: (N_i,)
        for i in xrange(P_nz):
            wAm_ = np.empty((Q, self.N[i]))
            for j in xrange(Q):
                A = self.compute_A(i, 1, g_gp_b[j], noise_parm)
                wAm_[j] = w[i, j] * A.dot(mg[j])
            Ah[i] = self.compute_A(i, h_gp_a[i], h_gp_b[i], noise_parm)
            y0[i] = y[i] - wAm_.sum(axis=0)

        Kh = np.empty(P_nz, dtype=object)
        Kh_inv = np.empty(P_nz, dtype=object)

        new_mh = self.mh.copy()
        new_Sh = self.Sh.copy()

        for i in xrange(P_nz):
            Kh[i], Kh_inv[i] = self.compute_K(h_gp_a[i], h_gp_b[i], noise_parm)
            A = Ah[i]
            Lambda = Kh_inv[i] + beta[i] * A.T.dot(A)

            new_Sh[i] = Si = inv(Lambda)
            new_mh[i] = Si.dot(beta[i] * A.T.dot(y0[i]))

        return new_mh, new_Sh

    def slfm_learn_h(self, var, check_grad=False):
        P, Q = self.P, self.Q
        P_nz = self.P_nz
        idx_nz = self.idx_nz
        M = self.M
        N = self.N
        x = self.x
        y = self.y
        z = self.z
        mg = self.mg
        Sh = self.Sh
        mh = self.mh

        w = var[:(P * Q)].reshape((P, Q))[idx_nz]
        var = var[(P * Q):]
        beta = var[:P][idx_nz]
        var = var[P:]
        g_gp_b = var[:Q]
        var = var[Q:]
        h_gp_a = var[:P][idx_nz]
        var = var[P:]
        h_gp_b = var[:P][idx_nz]

        #noise_parm = 1e-10
        #noise_parm = 0
        noise_parm = gp_noise

        #Am = np.empty(P, dtype=object)   # each entry: (N_i,)
        Ah = np.empty(P_nz, dtype=object)        # each entry: (N_i, M)
        # y0_i = y_i - A_i m_i - sum_j w_ij A_ij m_j
        y0 = np.empty(P_nz, dtype=object)        # each entry: (N_i,)
        for i in xrange(P_nz):
            wAm_ = np.empty((Q, self.N[i]))
            for j in xrange(Q):
                A = self.compute_A(i, 1, g_gp_b[j], noise_parm)
                wAm_[j] = w[i, j] * A.dot(mg[j])
            Ah[i] = self.compute_A(i, h_gp_a[i], h_gp_b[i], noise_parm)
            y0[i] = y[i] - wAm_.sum(axis=0)

        Kh = np.empty(P_nz, dtype=object)
        Kh_inv = np.empty(P_nz, dtype=object)

        for i in xrange(P_nz):
            Kh[i], Kh_inv[i] = self.compute_K(h_gp_a[i], h_gp_b[i], noise_parm)
            A = Ah[i]
            #Lambda = Kh_inv[i] + beta[i] * A.T.dot(A)

            #self.Sh[i] = Si = inv(Lambda)
            #self.mh[i] = Si.dot(beta[i] * A.T.dot(y0[i]))

        # compute evidence lower bound
        Sh = self.Sh
        mh = self.mh

        # derivative of kernel hyperparameters of h
        Knn = np.empty(P_nz, dtype=object)
        Kmn = np.empty(P_nz, dtype=object)
        Kmm = np.empty(P_nz, dtype=object)
        dKnn_da = np.empty(P_nz, dtype=object)
        dKnn_db = np.empty(P_nz, dtype=object)
        dKmn_da = np.empty(P_nz, dtype=object)
        dKmn_db = np.empty(P_nz, dtype=object)
        dKmm_da = np.empty(P_nz, dtype=object)
        dKmm_db = np.empty(P_nz, dtype=object)
        for i in xrange(P_nz):
            Kmm[i], dKmm_da[i], dKmm_db[i] = compute_dK(
                    z, z, h_gp_a[i], h_gp_b[i])
            Kmm[i] += np.eye(M) * noise_parm
            Knn[i], dKnn_da[i], dKnn_db[i] = compute_dK(
                    x[i], x[i], h_gp_a[i], h_gp_b[i])
            Kmn[i], dKmn_da[i], dKmn_db[i] = compute_dK(
                    z, x[i], h_gp_a[i], h_gp_b[i])

        dL_da = np.zeros(P_nz)
        dL_db = np.zeros(P_nz)
        for i in xrange(P_nz):
            Kmm_inv = inv(Kmm[i])
            A = Kmn[i].T.dot(Kmm_inv)
            dA_da = (dKmn_da[i].T - A.dot(dKmm_da[i])).dot(Kmm_inv)
            dA_db = (dKmn_db[i].T - A.dot(dKmm_db[i])).dot(Kmm_inv)
            b = beta[i]
            y_ = y0[i] - A.dot(mh[i])

            dL_da[i] = (-0.5 * np.trace(Kmm_inv.dot(dKmm_da[i]))
                        + 0.5 * np.trace(
                                Kmm_inv.dot(dKmm_da[i]).dot(Kmm_inv)
                                       .dot(Sh[i] + np.outer(mh[i], mh[i])))
                        + b * y_.T.dot(dA_da).dot(mh[i])
                        - (0.5 * b * np.trace(dKnn_da[i]
                                              - A.dot(dKmn_da[i])
                                              - dA_da.dot(Kmn[i])))
                        - b * np.trace(A.dot(Sh[i]).dot(dA_da.T)))

            dL_db[i] = (-0.5 * np.trace(Kmm_inv.dot(dKmm_db[i]))
                        + 0.5 * np.trace(
                                Kmm_inv.dot(dKmm_db[i]).dot(Kmm_inv)
                                       .dot(Sh[i] + np.outer(mh[i], mh[i])))
                        + b * y_.T.dot(dA_db).dot(mh[i])
                        - (0.5 * b * np.trace(dKnn_db[i]
                                              - A.dot(dKmn_db[i])
                                              - dA_db.dot(Kmn[i])))
                        - b * np.trace(A.dot(Sh[i]).dot(dA_db.T)))

        if regularize_h:
            dL_da -= 2 * self.shared_parms.reg_ah * h_gp_a

        d_h_gp_a = np.zeros(P)
        d_h_gp_b = np.zeros(P)
        d_h_gp_a[idx_nz] = dL_da
        d_h_gp_b[idx_nz] = dL_db

        return -np.concatenate((d_h_gp_a, d_h_gp_b))

    def predict(self, x_test):
        P = self.P
        Q = self.Q
        z = self.z
        P_nz = self.P_nz
        idx_nz = self.idx_nz

        L = len(x_test)
        mu_g = np.empty((Q, L))
        var_g = np.empty((Q, L))
        cov_g = np.empty((Q, L, L))
        mu_h = np.empty((P, L))
        var_h = np.empty((P, L))
        cov_h = np.empty((P, L, L))

        for j in xrange(Q):
            mu_g[j], var_g[j], cov_g[j] = gp_predict_approx(
                    1, self.shared_parms.g_gp_b[j],
                    self.mg[j], self.Sg[j], z, x_test)

        # Filling in default value when there is no observataion
        for i in xrange(P):
            if i not in idx_nz:
                mu_h[i] = 0
                ha = self.shared_parms.h_gp_a[i]
                hb = self.shared_parms.h_gp_b[i]
                var_h[i] = ha
                cov_h[i] = cov_mat(x_test, x_test, ha, hb)

        for i in xrange(P_nz):
            idx = idx_nz[i]
            mu_h[idx], var_h[idx], cov_h[idx] = gp_predict_approx(
                    self.shared_parms.h_gp_a[idx],
                    self.shared_parms.h_gp_b[idx],
                    self.mh[i], self.Sh[i], z, x_test)

        w = self.shared_parms.w
        mu = w.dot(mu_g) + mu_h
        cov = ((w**2)[..., np.newaxis, np.newaxis] * cov_g).sum(axis=1) + cov_h

        return mu, cov


class IndependentMultiOutputGP(object):
    def __init__(self, n_channels, n_latent_gps=5, n_inducing_points=20,
                 t_min=0, t_max=1):
        self.P = n_channels
        self.t_min = t_min
        self.t_max = t_max

    def train(self, raw_data):
        gp_parms = []
        for channel in xrange(self.P):
            non_empty = [ts[channel] for ts in raw_data
                         if len(ts[channel][0]) > 10]
            print len(non_empty)
            gp_parms.append(gp.learn_hyperparms(non_empty, mean_shift=False))
        self.gp_parms = gp_parms

    def save_model(self, pickle_name):
        pickle_save(pickle_name, self.gp_parms)

    def load_model(self, pickle_name):
        self.gp_parms, = pickle_load(pickle_name)

    def predictive_gaussian(self, test_raw, new_t):
        P = self.P
        L = len(new_t)
        mu = np.empty((P, L))
        cov = np.empty((P, L, L))
        for i in xrange(P):
            t_train, y_train = test_raw[i]
            gp_parms = self.gp_parms[i]
            mu[i], cov[i] = gp.posterior_mean_cov(
                    t_train, y_train, new_t, gp_parms, mean_shift=False)
        return mu, cov


class MultiOutputGP(object):
    def __init__(self, n_channels, n_latent_gps=5, n_inducing_points=20,
                 # w_reg_group options: 'none', 'individual', 'row', 'column'
                 w_reg_group='none',
                 # lasso parameter ('individual')
                 # the smaller the sparser of w
                 w_reg=1,
                 # l_{1,2} regularization parameter for w ('row')
                 # the larger the sparser of w
                 #w_reg=.5,
                 t_min=0, t_max=1):
        self.P = n_channels
        self.Q = n_latent_gps
        self.M = n_inducing_points
        self.t_min = t_min
        self.t_max = t_max
        self.w_reg_group = w_reg_group
        self.w_reg = w_reg

        self.z = np.linspace(t_min, t_max, self.M)
        self.shared = SharedParameters(self.z, self.P, self.Q)

    def train(self, data, maxiter=30):
        t1 = time.time()

        P = self.P
        Q = self.Q

        elbo_trace = []

        shared = self.shared

        g0 = np.concatenate((np.ravel(shared.w),
                             shared.beta,
                             np.ravel(shared.g_gp_b),
                             np.ravel(shared.h_gp_a),
                             np.ravel(shared.h_gp_b),
                             ))

        n_data = len(data)
        for n in xrange(n_data):
            data[n].update_m_S()

        bounds = ([(None, None)] * (P * Q) +
                  [(shared.beta_lbound, None)] * P +
                  [(shared.bg_lbound, None)] * Q +
                  [(shared.ah_lbound, None)] * P +
                  [(shared.bh_lbound, None)] * P)

        def f_df(g):
            elbo_sum = 0
            grad = np.zeros((P * Q +   # w
                             P +       # beta
                             Q +       # g_gp_b
                             P +       # h_gp_a
                             P         # h_gp_b
                            ))

            for n in xrange(n_data):
                elbo_sum += data[n].compute_elbo(g)
                grad += np.concatenate((data[n].slfm_learn(g),
                                        data[n].slfm_learn_h(g)))

            return elbo_sum, grad

        w_reg_group = self.w_reg_group
        w_reg = self.w_reg

        if w_reg_group == 'row':
            # penalize each row as a group
            groups = np.ravel(np.outer(np.arange(P), np.ones(Q, dtype=int)))
            n_groups = P

        if w_reg_group == 'column':
            # penalize each column as a group
            groups = np.ravel(np.outer(np.ones(P, dtype=int), np.arange(Q)))
            n_groups = Q

        if w_reg_group == 'row' or w_reg_group == 'column':
            orig_parms_len = len(g0)
            groupStart, groupPtr = groupl1_makeGroupPointers(groups)
            g0_augment = np.concatenate((g0, np.zeros(n_groups)))

            def penalized_f_df(parms):
                f, df = f_df(parms[:orig_parms_len])
                f += (w_reg * parms[orig_parms_len:]).sum()
                df = np.concatenate((df, w_reg * np.ones(n_groups)))
                return f, df

            def group_l12_project(parms):
                parms = parms.copy()
                w = parms[:(P * Q)]
                alpha = parms[orig_parms_len:]
                for i in xrange(n_groups):
                    group_idx = groupPtr[groupStart[i]:groupStart[i + 1]]
                    w[group_idx], alpha[i] = projectAux(w[group_idx], alpha[i])
                return box_project(parms, copy_parms=False)

        elif w_reg_group == 'individual':
            def lasso_project(parms):
                parms = parms.copy()
                len_w = P * Q
                w = parms[:len_w]
                parms[:len_w] = np.sign(w) * randomProject(np.fabs(w), w_reg)
                return box_project(parms, copy_parms=False)

        def box_project(parms, copy_parms=True):
            if copy_parms:
                parms = parms.copy()
            # Box constraints
            for i, (lbound, ubound) in enumerate(bounds):
                if lbound is not None and parms[i] < lbound:
                    parms[i] = lbound
                #if ubound is not None and parms[i] > ubound:
                #    parms[i] = ubound
            return parms

        for t in xrange(maxiter):
            print 't', t

            if w_reg_group == 'none':
                opt = fmin_l_bfgs_b(f_df, g0, bounds=bounds, maxiter=10,
                                    disp=0)[0]
                #opt = minConf_PQN(f_df, g0, box_project,
                #                  maxIter=20, verbose=0)[0]
            elif w_reg_group == 'row' or w_reg_group == 'column':
                #g0_augment = np.concatenate((g0, np.zeros(n_groups)))
                opt = minConf_PQN(penalized_f_df, g0_augment,
                                  group_l12_project, maxIter=20, verbose=0)[0]
                g0_augment = opt
                opt = opt[:orig_parms_len]
            elif w_reg_group == 'individual':
                opt = minConf_PQN(f_df, g0, lasso_project,
                                  maxIter=20, verbose=0)[0]

            g0_new = opt.copy()
            set_shared_parms(g0_new, shared)

            diff_norm = np.linalg.norm(g0_new - g0)
            print 'diff norm', diff_norm
            if diff_norm < .01:
                break
            g0 = g0_new
            print g0[:(P * Q)]

            diff_mg, diff_Sg, diff_mh, diff_Sh = [], [], [], []
            for n in xrange(n_data):
                old_mg, old_Sg = data[n].mg.copy(), data[n].Sg.copy()
                old_mh, old_Sh = data[n].mh.copy(), data[n].Sh.copy()
                data[n].update_m_S()
                new_mg, new_Sg = data[n].mg, data[n].Sg
                new_mh, new_Sh = data[n].mh, data[n].Sh

                diff_mg.append(np.linalg.norm(new_mg - old_mg))
                diff_Sg.append(np.linalg.norm(new_Sg - old_Sg))
                diff_mh.append(np.linalg.norm(new_mh - old_mh))
                diff_Sh.append(np.linalg.norm(new_Sh - old_Sh))

            print np.mean(diff_mg), np.mean(diff_Sg), np.mean(diff_mh), np.mean(diff_Sh)

            elbo_list = []
            for d in data:
                elbo_list.append(d.compute_elbo())

            #break

            mean_elbo = np.mean(elbo_list)
            print 'elbo', mean_elbo, np.std(elbo_list)

            elbo_trace.append(mean_elbo)

        t2 = time.time()
        print 'time', t2 - t1
        print self.shared.w

    def predictive_gaussian(self, time_series, new_t):
        time_series.update_m_S()
        return time_series.predict(new_t)

    def gen_collection(self, raw_data):
        collection = []
        for n, ts in enumerate(raw_data):
            x = [xy[0] for xy in ts]
            y = [xy[1] for xy in ts]
            collection.append(TimeSeries(x, y, self.shared))
        return collection

    def save_model(self, pickle_name):
        shared = self.shared
        pickle_save(pickle_name,
                    shared.w, shared.beta, shared.g_gp_b,
                    shared.h_gp_a, shared.h_gp_b)

    def load_model(self, pickle_name):
        w, beta, g_gp_b, h_gp_a, h_gp_b = pickle_load(pickle_name)
        self.shared.w = w
        self.shared.beta = beta
        self.shared.g_gp_b = g_gp_b
        self.shared.h_gp_a = h_gp_a
        self.shared.h_gp_b = h_gp_b


def main():
    np.random.seed(0)

    dat_id = 1
    ts_all, l_all = pickle_load('../chla-data/chla_ts_min0_%d.pkl' % dat_id)

    # randomly shuffle training examples
    idx = np.arange(len(ts_all))
    np.random.shuffle(idx)
    ts_all = ts_all[idx]
    #l_all = l_all[idx]

    parser = argparse.ArgumentParser()
    parser.add_argument('-q', dest='n_latent_gp', type=int, default=5)
    parser.add_argument('-g', dest='w_reg_group', default='none')
    parser.add_argument('-r', dest='w_reg', type=float, default=1)
    args = parser.parse_args()

    Q = args.n_latent_gp
    w_reg_group = args.w_reg_group
    w_reg = args.w_reg

    P = len(ts_all[0])
    #Q = 10
    M = 20

    #mogp = MultiOutputGP(P, Q, M)
    #mogp = MultiOutputGP(P, Q, M, w_reg_group='individual', w_reg=.5)
    #mogp = MultiOutputGP(P, Q, M, w_reg_group='row', w_reg=2)
    #mogp = MultiOutputGP(P, Q, M, w_reg_group='column', w_reg=1)
    #mogp = MultiOutputGP(P, Q, M, w_reg_group='column', w_reg=1.5)
    mogp = MultiOutputGP(P, Q, M, w_reg_group=w_reg_group, w_reg=w_reg)

    n_train = int(len(ts_all) * 0.5)
    #n_train = 8
    print 'n_train', n_train
    train_raw = ts_all[:n_train]
    train_ts = mogp.gen_collection(train_raw)
    mogp.train(train_ts, maxiter=50)

    w_reg_group_name = {
            'none': 'non',
            'row': 'row',
            'column': 'col',
            'individual': 'ind',
            }

    mogp_str = 'model-pqn-%s-%g-%d-%d' % (
            w_reg_group_name[mogp.w_reg_group], mogp.w_reg, Q, n_train)
    print mogp_str

    mogp_pickle = 'model/%s.pkl' % mogp_str
    mogp.load_model(mogp_pickle)
    test_raw = ts_all[n_train:]

    indep_gp = IndependentMultiOutputGP(P)
    indep_gp.train(train_raw)
    gp_parms = indep_gp.gp_parms

    for channel in xrange(P):
        loglike = []
        loglike_baseline = []
        for each_test in test_raw:
            x = [xy[0] for xy in each_test]
            y = [xy[1] for xy in each_test]
            channel_len = len(x[channel])
            if channel_len < 3:
                continue
            one_third = channel_len // 3
            x_held_out = x[channel][one_third:-one_third]
            y_held_out = y[channel][one_third:-one_third]
            x_remain = [each_x if i == channel else
                        np.concatenate((each_x[:one_third],
                                        each_x[-one_third:]))
                        for i, each_x in enumerate(x)]
            y_remain = [each_y if i == channel else
                        np.concatenate((each_y[:one_third],
                                        each_y[-one_third:]))
                        for i, each_y in enumerate(y)]
            ts = TimeSeries(x_remain, y_remain, mogp.shared)
            mu, cov = mogp.predictive_gaussian(ts, x_held_out)

            mean, var = gp.pointwise_posterior_mean_var(
                    x_remain[channel], y_remain[channel], x_held_out,
                    gp_parms[channel])

            for i, each_y in enumerate(y_held_out):
                loglike.append(norm.logpdf(each_y, mu[channel, i],
                                                   np.sqrt(cov[channel, i, i])))

                loglike_baseline.append(norm.logpdf(each_y, mean[i],
                                                    np.sqrt(var[i])))

        print '%2d %10.2f %10.2f %10.2f %10.2f %6d' % (
                channel, np.mean(loglike),
                np.std(loglike) / np.sqrt(len(loglike)),
                np.min(loglike), np.max(loglike), len(loglike))

        print '%2d %10.2f %10.2f %10.2f %10.2f %6d' % (
                channel, np.mean(loglike_baseline),
                np.std(loglike_baseline) / np.sqrt(len(loglike_baseline)),
                np.min(loglike_baseline), np.max(loglike_baseline),
                len(loglike_baseline))
        print '-' * 50
        pickle_save('loglike-cmp/%02d.pkl' % channel, loglike, loglike_baseline)
        pl.figure()
        pl.axis('equal')
        pl.scatter(loglike, loglike_baseline, alpha=.5)
        pl.savefig('loglike-cmp/loglike-%02d.pdf' % channel)
        pl.close()


def set_shared_parms(parms, shared):
    P, Q = shared.P, shared.Q
    shared.w = parms[:(P * Q)].reshape((P, Q))
    parms = parms[(P * Q):]
    shared.beta = parms[:P]
    parms = parms[P:]
    shared.g_gp_b = parms[:Q]
    parms = parms[Q:]
    shared.h_gp_a = parms[:P]
    parms = parms[P:]
    shared.h_gp_b = parms[:P]


if __name__ == '__main__':
    main()

