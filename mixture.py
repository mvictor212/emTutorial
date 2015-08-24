import numpy as np
from scipy import stats

class LnExponPMMixture(object):
    """
    A class for creating a mixture model with a log-normal, an exponential,
    and a varying number of point-mass components.
    """
    def __init__(self, pms=[]):
        """
        Parameters
        ----------
        pms : list
            List of point-masses for the model.
        """
        self.pms = pms
        self.K = len(pms) + 2

    def loglike(self, y):
        """
        Parameters
        ----------
        y : ndarray (N, )
            Array of positive real values.

        Returns
        -------
        logp : ndarray (N, K)
            A 2-d array, where entry i,k indicates likelihood of observation i coming from component k.
        """
        logp = np.zeros((len(y), self.K))
        logp[:, 0] = stats.lognorm.pdf(y + 0.001, scale=np.exp(self.mu), s=self.sigma) * self.weights[0]
        logp[:, 1] = stats.expon.pdf(y + 0.001, scale=self.lam) * self.weights[1]
        for i in range(self.K - 2):
            logp[:, i + 2] = (y == self.pms[i]) * self.weights[i + 2]
        logp = (logp.T / np.sum(logp, axis=1)).T
        return logp

    def m_step_mu(self, y, logp):
        """
        Update log-normal mean parameter.

        Parameters
        ----------
        y : ndarray (N, )
            Array of positive real values.

        logp : ndarray (N, K)
            A 2-d array, where entry i,k indicates likelihood of observation i coming from component k.
        """
        return np.dot(np.log(y + 0.001), logp) / np.sum(logp)

    def m_step_sigma(self, y, logp, mu):
        """
        Update log-normal variance parameter.

        Parameters
        ----------
        y : ndarray (N, )
            Array of positive real values.

        logp : ndarray (N, K)
            A 2-d array, where entry i,k indicates likelihood of observation i coming from component k.

        mu : float
            Log-normal mean.
        """
        return np.dot(logp, (np.log(y + 0.001) - self.mu) ** 2) / np.sum(logp)

    def m_step_lam(self, y, logp):
        """
        Update exponential parameter.

        Parameters
        ----------
        y : ndarray (N, )
            Array of positive real values.

        logp : ndarray (N, K)
            A 2-d array, where entry i,k indicates likelihood of observation i coming from component k.
        """
        return np.dot(y, logp) / np.sum(logp)

    def m_step_weights(self, logp):
        """
        Update weights over components.

        Parameters
        ----------
        logp : ndarray (N, K)
            A 2-d array, where entry i,k indicates likelihood of observation i coming from component k.
        """
        return np.sum(logp, axis=0) / len(logp)

    def m_step(self, y, logp):
        """
        Update all parameters.

        Parameters
        ----------
        y : ndarray (N, )
            Array of positive real values.

        logp : ndarray (N, K)
            A 2-d array, where entry i,k indicates likelihood of observation i coming from component k.
        """
        self.mu = self.m_step_mu(y, logp[:, 0])
        self.sigma = np.sqrt(self.m_step_sigma(y, logp[:, 0], self.mu))
        self.lam = self.m_step_lam(y, logp[:, 1])
        self.weights = self.m_step_weights(logp)

    def fit(self, y, max_iter=100):
        """
        Find MLE of stipulated model.

        Parameters
        ----------
        y : ndarray (N, )
            Array of positive real values.

        max_iter : int
            Maximum number of iterations allowed in EM algorithm.
        """
        self.mu = np.mean(np.log(y + 0.001))
        self.sigma = 1.0
        self.lam = 10.0,
        weights = np.ones(self.K) + np.random.uniform(0., 0.1, size=self.K)
        self.weights = weights / np.sum(weights)
        for i in range(max_iter):
            logp = self.loglike(y)
            self.m_step(y, logp)
            if self.sanity_check():
                break
        return self

    def sanity_check(self):
        """
        Perform sanity check, requiring exponential parameter to be positive.
        """
        if self.lam < 1e-5:
            self.pms.append(0.0)
            self.K += 1
            self.weights.resize(self.K)
            self.weights[-1] = self.weights[1]
            self.weights[1] = 0
            self.lam = 1e-4
            return True
        else:
            return False

    def _draw(self):
        """
        Draw single component, value combo from fitted model.
        """
        comp = np.argmax(np.random.multinomial(1, self.weights))
        if comp == 0:
            drw = np.random.lognormal(self.mu, self.sigma)
        elif comp == 1:
            drw = np.random.exponential(scale=self.lam)
        else:
            drw = self.pms[comp - 2]
        return comp, drw

    def sample(self, size=100):
        """
        Draw `size` component/value samples from fitted model.
        """
        comps = []
        drws = []
        for _ in range(size):
            comp, drw = self._draw()
            comps.append(comp)
            drws.append(drw)
        return comps, drws
