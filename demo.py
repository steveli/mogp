import numpy as np
import pylab as pl
import time
#from multi_output_gp import MultiOutputGP
from elbo_pqn import MultiOutputGP
from pickle_io import pickle_save, pickle_load


class TimeSeriesMaker(object):

    def __init__(self, n_ts):
        self.min_x, self.max_x = 0, 1
        self.n_ts = n_ts

        self.fig, self.ax = pl.subplots(n_ts, 1, figsize=(9, 9))
        self.fig.canvas.mpl_connect('button_press_event', self.onmousepress)
        self.fig.canvas.mpl_connect('key_press_event', self.onpress)
        self.ax_idx = {}
        for idx, each_ax in enumerate(self.ax):
            self.ax_idx[each_ax] = idx
            self.clear_ax(each_ax)

        try:
            self.x, self.y = pickle_load('input.pkl')
            for idx, each_ax in enumerate(self.ax):
                self.ax[idx].plot(self.x[idx], self.y[idx], 'ko')
        except:
            self.reset_data()

        pl.show()

    def reset_data(self):
        self.x = [[] for i in xrange(self.n_ts)]
        self.y = [[] for i in xrange(self.n_ts)]

    def onmousepress(self, event):
        ax = event.inaxes
        if ax:
            idx = self.ax_idx[ax]
            if event.button == 1:
                x, y = event.xdata, event.ydata
                self.x[idx].append(x)
                self.y[idx].append(y)
                ax.plot(x, y, 'ko')
            event.canvas.draw()

    def clear_ax(self, ax):
        ax.cla()
        ax.set_xlim(self.min_x, self.max_x)
        ax.set_ylim(-3, 3)

    def onpress(self, event):
        if event.key == 'd':
            mtgp = MultiOutputGP(n_channels=3, n_latent_gps=2)
            t1 = time.time()
            raw_ts = [zip(self.x, self.y)]
            train_ts = mtgp.gen_collection(raw_ts)
            mtgp.train(train_ts, maxiter=50)
            t2 = time.time()
            print 'time:', t2 - t1
            #test_x = [np.linspace(self.min_x, self.max_x, 100)] * self.n_ts
            test_x = np.linspace(self.min_x, self.max_x, 100)
            #test_x = [[1, 2], [2, 3]]
            #post_mean, post_cov = mtgp.predict(test_x)
            post_mean, post_cov = mtgp.predictive_gaussian(train_ts[0], test_x)
            for i in xrange(self.n_ts):
                #std = np.sqrt(np.diag(post_cov[i]))
                std = np.sqrt(np.diag(post_cov[i]))
                ax = self.ax[i]
                self.clear_ax(ax)
                ax.fill_between(test_x,
                                post_mean[i] - std,
                                post_mean[i] + std,
                                edgecolor='none', alpha=.3, color='g')
                ax.plot(test_x, post_mean[i], 'g-')
                ax.plot(self.x[i], self.y[i], 'ko')
            self.fig.canvas.draw()
        elif event.key == 'm':
            pickle_save('input.pkl', self.x, self.y)
        elif event.key == 'c':
            self.reset_data()
            for ax in self.ax:
                self.clear_ax(ax)
            self.fig.canvas.draw()


def main():
    np.random.seed(0)
    TimeSeriesMaker(3)


if __name__ == '__main__':
    main()
