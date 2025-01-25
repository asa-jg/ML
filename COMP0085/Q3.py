import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

class VariationalEM:
    def __init__(self, x, k, iterations):
        self.x = x
        self.k = k
        self.iterations = iterations
        self.n, self.d = x.shape
        self.fList = []
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
        print(self.sigma)
        self.lambdaOld = lambdaInit

    @staticmethod
    def meanField(x, mu, sigma, pi, lambda_old, max_steps):
        """
        Perform mean-field optimization to estimate parameters.

        Parameters:
            x (np.ndarray): Input data matrix of shape (n, d).
            mu (np.ndarray): Mean parameters of shape (d, k).
            sigma (float): Standard deviation parameter (assumed constant across dimensions).
            pi (np.ndarray): Prior probabilities of shape (1, k).
            lambda_old (np.ndarray): Initial responsibility matrix of shape (n, k).
            max_steps (int): Maximum number of optimization steps.

        Returns:
            tuple: Updated responsibility matrix (lambda_new) and final objective value (f).
        """
        n, d = x.shape
        k = pi.shape[1]
        pop = 1e-20
        current_lambda = lambda_old
        F = -np.inf

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

            # if prevF > F:
            #     print(f"Warning: F decreased by {prevF - F}")

            if F - prevF < pop:
                return current_lambda, F

        return current_lambda, F

    @staticmethod
    def mStep(X, ES, ESS):
        """
        mu, sigma, pie = MStep(X,ES,ESS)

        Inputs:
        -----------------
               X: shape (N, D) data matrix
              ES: shape (N, K) E_q[s]
             ESS: shape (K, K) sum over data points of E_q[ss'] (N, K, K)
                               if E_q[ss'] is provided, the sum over N is done for you.

        Outputs:
        --------
              mu: shape (D, K) matrix of means in p(y|{s_i},mu,sigma)
           sigma: shape (,)    standard deviation in same
             pie: shape (1, K) vector of parameters specifying generative distribution for s
        """
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

    def run(self):
        self.initialiseParams()
        fOld = -np.inf

        for i in range(self.iterations):
            lambdaNew, f = self.meanField(self.x, self.mu, self.sigma, self.pi, self.lambdaOld, 10)

            es = lambdaNew
            ess = np.zeros((self.k, self.k))
            for n in range(self.n):
                matrix = np.outer(lambdaNew[n, :], lambdaNew[n, :])
                np.fill_diagonal(matrix, lambdaNew[n, :])
                ess += matrix

            self.mu, self.sigma, self.pi = self.mStep(self.x, es, ess)
            self.lambdaOld = lambdaNew

            _, f = self.meanField(self.x, self.mu, self.sigma, self.pi, self.lambdaOld, 50)
            self.fList.append(f)

            if f - fOld < self.threshold:
                np.savetxt('lambda.txt', self.lambdaOld)
                return self.mu, self.sigma, self.pi, self.fList

            fOld = f

        np.savetxt('lambda.txt', self.lambdaOld)
        return self.mu, self.sigma, self.pi, self.fList


def plot_log_free_energy_differences(fList, output_path="log_free_energy_diff.pdf"):
    """
    Plot log differences of free energy values over iterations.

    Parameters:
        fList (list or np.ndarray): List of free energy values over iterations.
        output_path (str): Path to save the resulting plot.
    """
    f_differences = np.diff(fList)

    f_differences_clipped = np.clip(f_differences, 1e-15, None)

    log_f_differences = np.log(f_differences_clipped)
    log_f_differences
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(log_f_differences)), log_f_differences[:-1], marker='o')
    plt.xlabel("Iteration")
    plt.ylabel("log(F(t) - F(t-1))")
    plt.title("Log Differences of Free Energy Over Iterations")
    plt.tight_layout()
    plt.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.show()


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

    variationalEm = VariationalEM(y, k, iterations)
    mu, sigma, pi, fList = variationalEm.run()

    plt.figure()
    plt.plot(range(1, len(fList) + 1), fList)
    plt.xlabel("{Iteration}")
    plt.ylabel("{Free Energy}")
    plt.title("{Variational Free Energy Convergence}")
    plt.show()

    fig, axs = plt.subplots(1, k)
    fig.suptitle(f"\textbf{{Features for K = {k}}}")
    for i in range(k):
        axs[i].imshow(np.reshape(mu[:, i], (4, 4)), cmap='gray')
        axs[i].axis('off')
    plt.savefig("3f_features.pdf", format="pdf", bbox_inches="tight")
    plt.show()
    plot_log_free_energy_differences(fList, output_path="log_free_energy_diff.pdf")
