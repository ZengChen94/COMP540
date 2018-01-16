import numpy as np
import matplotlib.pyplot as plt


class ProbabilityModel:
    # Returns a single sample (independent of values returned on previous calls).
    # The returned value is an element of the model's sample space.
    def sample(self):
        pass


# The sample space of this probability model is the set of real numbers, and
# the probability measure is defined by the density function 
# p(x) = 1/(sigma * (2*pi)^(1/2)) * exp(-(x-mu)^2/2*sigma^2)
class UnivariateNormal(ProbabilityModel):
    # Initializes a univariate normal probability model object
    # parameterized by mu and (a positive) sigma
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    # Box-Muller transform
    def sample(self):
        u1 = np.random.uniform(0, 1.0)
        u2 = np.random.uniform(0, 1.0)
        z1 = np.sqrt(-2.0 * np.log(u1)) * np.cos(2 * np.pi * u2)
        return z1 * self.sigma + self.mu


# The sample space of this probability model is the set of D dimensional real
# column vectors (modeled as numpy.array of size D x 1), and the probability 
# measure is defined by the density function 
# p(x) = 1/(det(Sigma)^(1/2) * (2*pi)^(D/2)) * exp( -(1/2) * (x-mu)^T * Sigma^-1 * (x-mu) )
class MultiVariateNormal(ProbabilityModel):
    # Initializes a multivariate normal probability model object
    # parameterized by Mu (numpy.array of size D x 1) expectation vector 
    # and symmetric positive definite covariance Sigma (numpy.array of size D x D)
    def __init__(self, Mu, Sigma):
        self.Mu = Mu
        self.Sigma = Sigma

    # Cholesky method
    def sample(self):
        L = np.linalg.cholesky(self.Sigma)
        M = []
        for i in range(len(self.Mu)):
            M.append(UnivariateNormal(0, 1.0).sample())
        M = np.array(M)
        return self.Mu + np.dot(L, M)


# The sample space of this probability model is the finite discrete set {0..k-1}, and 
# the probability measure is defined by the atomic probabilities 
# P(i) = ap[i]
class Categorical(ProbabilityModel):
    # Initializes a categorical (a.k.a. multinom, multinoulli, finite discrete)
    # probability model object with distribution parameterized by the atomic probabilities vector
    # ap (numpy.array of size k).
    def __init__(self, ap):
        self.ap = ap

    def sample(self):
        p = np.random.uniform(0, 1.0)
        sum = 0
        for i in range(len(self.ap)):
            if sum <= p < sum + self.ap[i]:
                return i
            sum += self.ap[i]
        return 0


# The sample space of this probability model is the union of the sample spaces of
# the underlying probability models, and the probability measure is defined by 
# the atomic probability vector and the densities of the supplied probability models
# p(x) = sum ad[i] p_i(x)
class MixtureModel(ProbabilityModel):
    # Initializes a mixture-model object parameterized by the
    # atomic probabilities vector ap (numpy.array of size k) and by the tuple of 
    # probability models pm
    def __init__(self, ap, pm):
        self.ap = ap
        self.pm = pm

    def sample(self):
        p = np.random.uniform(0, 1.0)
        sum = 0
        for i in range(len(self.ap)):
            if sum <= p < sum + self.ap[i]:
                return self.pm[i].sample()
            sum += self.ap[i]
        return 0


# Categorical Distribution with Probabilities [0.2,0.4,0.3,0.1]
def histogramOfCategoricalDistribution():
    probabilities = [0.2, 0.4, 0.3, 0.1]
    samples = []
    for i in range(10000):
        samples.append(Categorical(probabilities).sample())

    count, bins, ignored = plt.hist(samples, 15, normed=True)
    plt.xlim((0, 3))
    plt.xlabel('Categorical Distribution with Probabilities [0.2,0.4,0.3,0.1]')
    plt.plot(bins, np.ones_like(bins), linewidth=2, color='r')
    plt.show()


# Univariate Normal Distribution with Mean of 10 and Standard Deviation 1
def histogramOfUnivariateNormalDistribution():
    mu = 10
    sigma = 1
    samples = []
    for i in range(10000):
        samples.append(UnivariateNormal(mu, sigma).sample())
    count, bins, ignored = plt.hist(samples, 50, normed=True)
    plt.xlabel('Univariate Normal Distribution with Mean of 10 and Standard Deviation 1')
    plt.plot(bins, 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(bins - mu) ** 2 / (2 * sigma ** 2)), linewidth=2,
             color='r')
    plt.show()


# 2-D Gaussian with Mean at [1, 1] and a covariance matrix [[1, 0.5], [0.5, 1]]
def scatterPlotOfMultiVariateNormalDistribution():
    Mu = [1, 1]
    Sigma = [[1, 0.5], [0.5, 1]]
    x = []
    y = []
    for i in range(1000):
        sample = MultiVariateNormal(Mu, Sigma).sample()
        x.append(sample[0])
        y.append(sample[1])
    plt.scatter(x, y)
    plt.xlabel('2-D Gaussian with Mean at [1, 1] and a covariance matrix [[1, 0.5], [0.5, 1]]')
    plt.axis('equal')
    plt.show()


# Test the mixture sampling code
def testMixtureSampling():
    I = np.diag([1, 1])
    Mu = [[1, 1], [1, -1], [-1, 1], [-1, -1]]
    n = 4
    pm = []
    ap = []
    for i in range(n):
        pm.append(MultiVariateNormal(Mu[i], I))
        ap.append(1.0 / n)

    center = [0.1, 0.2]
    total = 20000
    count = 0
    radius = 1
    for i in range(total):
        sample = MixtureModel(ap, pm).sample()
        if np.linalg.norm(sample - center) <= radius:
            count += 1

    probability = 1.0 * count / total
    print 'The Probability is around', probability


def main():
    # histogramOfCategoricalDistribution()
    # histogramOfUnivariateNormalDistribution()
    # scatterPlotOfMultiVariateNormalDistribution()
    testMixtureSampling()


if __name__ == '__main__':
    main()
