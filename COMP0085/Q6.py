import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc


class VariationalEM:
    def __init__(self, x, k, iterations):
        self.x = x
        self.k = k
        self.iterations = iterations
        self.n, self.d = x.shape

        # For logging free energy over outer iterations
        self.fList = []
        self.threshold = 1e-20

        # Model parameters
        self.mu = None
        self.sigma = None
        self.pi = None

        # "Responsibilities" or posterior means for s_i
        self.lambdaOld = None

        # Loopy BP messages, shape: (K, K, N)
        # We will initialize these in initialiseParams().
        self.messages = None

    def initialiseParams(self):
        """
        Randomly initialize parameters and messages.
        """
        # 1) Initialize lambda (responsibilities)
        lambdaInit = np.random.rand(self.n, self.k)
        lambdaInit = np.clip(lambdaInit, 1e-15, 1 - 1e-15)

        # 2) Build ES and ESS
        es = lambdaInit
        ess = np.zeros((self.k, self.k))
        for n_idx in range(self.n):
            matrix = np.outer(lambdaInit[n_idx, :], lambdaInit[n_idx, :])
            # Diagonal replaced by s_i (not s_i^2) if we are modeling Bernoulli s_i
            np.fill_diagonal(matrix, lambdaInit[n_idx, :])
            ess += matrix

        # 3) M-step to initialize mu, sigma, pi
        self.mu, self.sigma, self.pi = self.mStep(self.x, es, ess)

        # 4) Store the current responsibilities
        self.lambdaOld = lambdaInit

        # 5) Initialize loopy BP messages
        # Shape (K, K, N) for pairwise messages i->j for each data point
        self.messages = np.random.rand(self.k, self.k, self.n)
        # By convention, set diagonal = 0.0 (message from i->i)
        for data_idx in range(self.n):
            np.fill_diagonal(self.messages[:, :, data_idx], 0.0)

        print(f"Initial sigma: {self.sigma}")

    def run(self):
        """
        Main Variational EM loop, replacing the E-step with loopy BP.
        """
        self.initialiseParams()
        fOld = -np.inf

        for i in range(self.iterations):
            # ---------- E-step via Loopy BP -----------
            lambdaNew, fVal = self.loopyBPEStep(
                self.x, self.mu, self.sigma, self.pi,
                self.lambdaOld, self.messages, max_steps=10
            )
            # ------------------------------------------

            # Construct ES, ESS for M-step
            es = lambdaNew
            ess = np.zeros((self.k, self.k))
            for n_idx in range(self.n):
                matrix = np.outer(lambdaNew[n_idx, :], lambdaNew[n_idx, :])
                np.fill_diagonal(matrix, lambdaNew[n_idx, :])
                ess += matrix

            # M-step
            self.mu, self.sigma, self.pi = self.mStep(self.x, es, ess)
            self.lambdaOld = lambdaNew

            # Optional: compute free energy again (or do a final E-step for F)
            _, fVal = self.loopyBPEStep(
                self.x, self.mu, self.sigma, self.pi,
                self.lambdaOld, self.messages, max_steps=50,
                compute_free_energy=True
            )

            self.fList.append(fVal)

            # Check for convergence
            if (fVal - fOld) < self.threshold:
                # Save final responsibilities if desired
                np.savetxt('lambda.txt', self.lambdaOld)
                return self.mu, self.sigma, self.pi, self.fList

            fOld = fVal

        np.savetxt('lambda.txt', self.lambdaOld)
        return self.mu, self.sigma, self.pi, self.fList

    @staticmethod
    def mStep(X, ES, ESS):
        """
        mu, sigma, pie = MStep(X, ES, ESS)

        Inputs:
        -----------------
               X:   shape (N, D) data matrix
              ES:   shape (N, K) E_q[s]
             ESS:   shape (K, K) sum over data points of E_q[ss']
                    (or shape (N, K, K) if not summed)

        Outputs:
        --------
              mu: shape (D, K) means in p(x|s, mu, sigma)
           sigma: scalar for isotropic covariance
             pie: shape (1, K) prior probabilities for s
        """
        N, D = X.shape
        if ES.shape[0] != N:
            raise TypeError('ES must have the same number of rows as X')
        K = ES.shape[1]
        if ESS.shape == (N, K, K):
            ESS = np.sum(ESS, axis=0)
        if ESS.shape != (K, K):
            raise TypeError('ESS must be (K, K) after summing')

        # Solve for mu
        mu = np.linalg.inv(ESS) @ ES.T @ X
        mu = mu.T  # shape: (D, K)

        # Solve for sigma
        numerator = (
                np.trace(X.T @ X)
                + np.trace(mu.T @ mu @ ESS)
                - 2 * np.trace(ES.T @ X @ mu)
        )
        sigma = np.sqrt(numerator / (N * D))

        # Solve for pi
        pie = np.mean(ES, axis=0, keepdims=True)

        return mu, sigma, pie

    def loopyBPEStep(self, x, mu, sigma, pi, lambdaOld, messages, max_steps=50,
                     compute_free_energy=False):
        """
        Loopy BP / EP-style E-step.

        Parameters
        ----------
        x :        (N, D) data
        mu :       (D, K)
        sigma :    float, isotropic std
        pi :       (1, K) or (K,)
        lambdaOld: (N, K) previous iteration's responsibilities
        messages:  (K, K, N) messages between latent factors for each datapoint
        max_steps: int, number of loopy BP iterations
        compute_free_energy: bool, if True, compute an approximate free energy at the end

        Returns
        -------
        lambdaNew : (N, K) updated responsibilities
        fVal      : float,  approximate free energy (if computed, else 0.0)
        """
        N, D = x.shape
        if pi.shape[0] == 1:
            pi = pi.flatten()
        K = len(pi)

        logit_pi = np.log(np.clip(pi, 1e-15, 1 - 1e-15) / (1 - np.clip(pi, 1e-15, 1 - 1e-15)))

        diagMuMu = np.diag(mu.T @ mu)  # shape: (K,)

        fLocal = np.zeros((N, K))
        for i in range(N):
            fLocal[i, :] = (
                    (x[i, :] @ mu) / (sigma ** 2)
                    + logit_pi
                    - diagMuMu / (2.0 * sigma ** 2)
            )

        threshold = 1e-15
        newMessages = messages.copy()

        for _ in range(max_steps):
            oldMessages = newMessages.copy()
            for n_idx in range(N):
                for i_idx in range(K):
                    for j_idx in range(i_idx + 1, K):
                        wVal = -(mu[:, i_idx].T @ mu[:, j_idx]) / (sigma ** 2)

                        sumOverOthers_i = np.sum(newMessages[:, i_idx, n_idx]) - newMessages[j_idx, i_idx, n_idx]
                        otherTerms_i = fLocal[n_idx, i_idx] + sumOverOthers_i

                        ratio = (1 + np.exp(otherTerms_i + wVal)) / (1 + np.exp(otherTerms_i))
                        newMessages[i_idx, j_idx, n_idx] = 0.5 * np.log(ratio) \
                                                           + 0.5 * newMessages[i_idx, j_idx, n_idx]

                        wVal2 = -(mu[:, j_idx].T @ mu[:, i_idx]) / (sigma ** 2)
                        sumOverOthers_j = np.sum(newMessages[:, j_idx, n_idx]) - newMessages[i_idx, j_idx, n_idx]
                        otherTerms_j = fLocal[n_idx, i_idx] + sumOverOthers_j

                        ratio2 = (1 + np.exp(otherTerms_j + wVal2)) / (1 + np.exp(otherTerms_j))
                        newMessages[j_idx, i_idx, n_idx] = np.log(ratio2) \
                                                           + 0.5 * newMessages[j_idx, i_idx, n_idx]

            lambdaNew = np.zeros((N, K))
            for n_idx in range(N):
                sumMsg = np.sum(newMessages[:, :, n_idx], axis=0)  # shape (K,)

                lambdaNew[n_idx, :] = 1.0 / (1.0 + np.exp(-(fLocal[n_idx, :] + sumMsg)))

            lambdaNew = np.clip(lambdaNew, 1e-15, 1 - 1e-15)

            if np.max(np.abs(oldMessages - newMessages)) < threshold:
                break

        fVal = 0.0
        if compute_free_energy:

            fVal = self._approxFreeEnergy(x, mu, sigma, pi, lambdaNew)

            return lambdaNew, fVal

    def _approxFreeEnergy(self, x, mu, sigma, pi, lambdaVals):
        """
        Roughly compute a free-energy-like objective for debugging.

        This is optional. Adapt from the original code or use
        your own approximation under the EP assumptions.
        """
        n, d = x.shape
        k = len(pi)

        # Just reuse the old formula from meanField as a placeholder
        # (This is not a rigorous EP free energy, but an indicative quantity.)
        F = (
                -np.sum(np.square(x)) / (2 * sigma ** 2)
                + np.sum((x @ mu) * lambdaVals) / (sigma ** 2)
                - (1.0 / (2 * sigma ** 2)) * (
                        np.sum((lambdaVals @ mu.T) ** 2)
                        - np.sum(np.diag(mu.T @ mu) * (lambdaVals ** 2).sum(axis=0))
                        + np.sum(np.diag(mu.T @ mu) * lambdaVals.sum(axis=0))
                )
                - n * d * np.log(2 * np.pi * sigma ** 2) / 2
                + np.sum(
            lambdaVals * np.log(np.clip(pi, 1e-15, 1.0))
            + (1 - lambdaVals) * np.log(np.clip(1 - pi, 1e-15, 1.0))
        )
                - np.sum(
            lambdaVals * np.log(np.clip(lambdaVals, 1e-15, 1.0))
            + (1 - lambdaVals) * np.log(np.clip(1 - lambdaVals, 1e-15, 1.0))
        )
        )
        return F


def plot_log_free_energy_differences(fList, output_path="log_free_energy_diff.pdf"):
    """
    Plot log differences of free energy values over iterations.
    """
    f_differences = np.diff(fList)
    f_differences_clipped = np.clip(f_differences, 1e-15, None)
    log_f_differences = np.log(f_differences_clipped)

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(log_f_differences) + 1), log_f_differences, marker='o')
    plt.xlabel("Iteration")
    plt.ylabel("log(F(t) - F(t-1))")
    plt.title("Log Differences of Free Energy Over Iterations")
    plt.tight_layout()
    plt.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    rc('text', usetex=True)
    rc('font', family='serif')

    # Example usage
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

    # Plot free energy over iterations
    plt.figure()
    plt.plot(range(1, len(fList) + 1), fList)
    plt.xlabel("Iteration")
    plt.ylabel("Free Energy")
    plt.title("Variational Free Energy (Loopy BP E-step)")
    plt.savefig("3f_free_energy_loopyBP.pdf", format="pdf", bbox_inches="tight")
    plt.show()

    fig, axs = plt.subplots(1, k, figsize=(2 * k, 2))
    fig.suptitle(f"Features for K = {k}")
    for i in range(k):
        axs[i].imshow(np.reshape(mu[:, i], (4, 4)), cmap='gray')
        axs[i].axis('off')
    plt.savefig("3f_features_loopyBP.pdf", format="pdf", bbox_inches="tight")
    plt.show()

    if len(fList) > 1:
        plot_log_free_energy_differences(fList, output_path="log_free_energy_diff_loopyBP.pdf")
