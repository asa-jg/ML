import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

class VariationalEM:
    def __init__(self, x, k, iterations):
        self.x = x
        self.k = k
        self.iterations = iterations
        self.n, self.d = x.shape
        self.threshold = 1e-20
        self.mu = None
        self.sigma = None
        self.pi = None
        self.lambdaOld = None

    def initialiseParams(self):
        lambdaInit = np.random.rand(self.n, self.k)
        lambdaInit = np.clip(lambdaInit, 1e-15, 1 - 1e-15)
        es = lambdaInit
        ess = np.zeros((self.k, self.k))

        for n in range(self.n):
            matrix = np.outer(lambdaInit[n, :], lambdaInit[n, :])
            np.fill_diagonal(matrix, lambdaInit[n, :])
            ess += matrix

        self.mu, self.sigma, self.pi = self.mStep(self.x, es, ess)
        self.lambdaOld = lambdaInit

    @staticmethod
    def meanField(x, mu, sigma, pi, lambda_old, max_steps):
        n, d = x.shape
        k = pi.shape[1]
        threshold = 1e-20
        current_lambda = lambda_old
        F = -np.inf
        free_energy_history = []

        for step in range(max_steps):
            prevF = F

            current_lambda = 1 / (1 + np.exp(-(
                    x @ mu / (sigma ** 2) +
                    -0.5 * np.sum(mu ** 2, axis=0) / (sigma ** 2) +
                    -((current_lambda @ (mu.T @ mu)) / (sigma ** 2)) +
                    (np.multiply(current_lambda, np.sum(mu ** 2, axis=0)) / (sigma ** 2)) +
                    np.log(pi / (1 - pi))
            )))
            current_lambda = np.clip(current_lambda, 1e-15, 1 - 1e-15)

            F = (
                    -np.sum(np.square(x)) / (2 * sigma ** 2) +
                    np.sum(np.sum(np.multiply(x @ mu, current_lambda), axis=1)) / (sigma ** 2) +
                    (-1 / (2 * sigma ** 2)) * (
                            np.sum(np.sum((current_lambda @ mu.T) ** 2)) +
                            -np.sum(np.multiply(np.tile(np.diag(mu.T @ mu), (n, 1)), np.square(current_lambda))) +
                            np.sum(np.multiply(np.tile(np.diag(mu.T @ mu), (n, 1)), current_lambda))
                    ) +
                    -n * d * np.log(2 * np.pi * (sigma ** 2)) / 2 +
                    np.sum(
                        np.multiply(current_lambda, np.log(pi)) +
                        np.multiply(1 - current_lambda, np.log(1 - pi))
                    ) +
                    -np.sum(
                        np.multiply(current_lambda, np.log(current_lambda + 1e-15)) +
                        np.multiply((1 - current_lambda), np.log(1 - current_lambda + 1e-15))
                    )
            )
            free_energy_history.append(F)

            if F - prevF < threshold:
                break

        return current_lambda, free_energy_history

    @staticmethod
    def mStep(X, ES, ESS):
        N, D = X.shape
        if ES.shape[0] != N:
            raise TypeError('ES must have the same number of rows as X')
        K = ES.shape[1]
        if ESS.shape == (N, K, K):
            ESS = np.sum(ESS, axis=0)
        if ESS.shape != (K, K):
            raise TypeError('ESS must be square and have the same number of columns as ES')

        mu = np.dot(np.dot(np.linalg.inv(ESS), ES.T), X).T
        sigma = np.sqrt((np.trace(np.dot(X.T, X)) + np.trace(np.dot(np.dot(mu.T, mu), ESS))
                         - 2 * np.trace(np.dot(np.dot(ES.T, X), mu))) / (N * D))
        pie = np.mean(ES, axis=0, keepdims=True)

        return mu, sigma, pie


if __name__ == "__main__":
    rc('text', usetex=True)
    rc('font', family='serif')

    k = 8
    iterations = 100

    features = [
        [0, 0, 1, 0,
         0, 1, 1, 1,
         0, 0, 1, 0,
         0, 0, 0, 0],
        [0, 1, 0, 0,
         0, 1, 0, 0,
         0, 1, 0, 0,
         0, 1, 0, 0],
        [1, 1, 1, 1,
         0, 0, 0, 0,
         0, 0, 0, 0,
         0, 0, 0, 0],
        [1, 0, 0, 0,
         0, 1, 0, 0,
         0, 0, 1, 0,
         0, 0, 0, 1],
        [0, 0, 0, 0,
         0, 0, 0, 0,
         1, 1, 0, 0,
         1, 1, 0, 0],
        [1, 1, 1, 1,
         1, 0, 0, 1,
         1, 0, 0, 1,
         1, 1, 1, 1],
        [0, 0, 0, 0,
         0, 1, 1, 0,
         0, 1, 1, 0,
         0, 0, 0, 0],
        [0, 0, 0, 1,
         0, 0, 0, 1,
         0, 0, 0, 1,
         0, 0, 0, 1],
    ]

    numSamples = 2000
    numFeatures = 16
    numFactors = len(features)

    weights = 0.5 + np.random.rand(numFactors, 1) * 0.5
    mu = np.array([w * f for w, f in zip(weights, features)])
    latentFactors = np.random.rand(numSamples, numFactors) < 0.3
    data = latentFactors @ mu + np.random.randn(numSamples, numFeatures) * 0.1

    y = data
    first_data_point = y[:1, :]

    variationalEm = VariationalEM(y, k, iterations)
    variationalEm.initialiseParams()

    sigmas = [1, 0.1, 10]
    free_energy_histories = []

    for sigma in sigmas:
        lambda_init = np.random.rand(1, k)
        lambda_init = np.clip(lambda_init, 1e-15, 1 - 1e-15)
        _, free_energy_history = variationalEm.meanField(first_data_point, variationalEm.mu, sigma, variationalEm.pi, lambda_init, 50)
        free_energy_histories.append(free_energy_history)

    plt.figure(figsize=(12, 6))
    for i, sigma in enumerate(sigmas):
        plt.plot(free_energy_histories[i], label=f"Sigma = {sigma}")
    plt.xlabel("Iteration")
    plt.ylabel("Free Energy (F)")
    plt.title("Free Energy Convergence for Different Sigmas")
    plt.legend()
    plt.show()

    plt.figure(figsize=(12, 6))
    for i, sigma in enumerate(sigmas):
        log_diff = np.log(np.diff(free_energy_histories[i]) + 1e-15)
        plt.plot(log_diff, label=f"Sigma = {sigma}")
    plt.xlabel("Iteration")
    plt.ylabel("Log Difference of Free Energy (log(F(t) - F(t-1)))")
    plt.title("Log Differences in Free Energy for Different Sigmas")
    plt.legend()
    plt.show()
