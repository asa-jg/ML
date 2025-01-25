import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


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


def EP_E_step(X, mu, sigma, pi, messages, iterations):
    """
    Runs the EP (Expectation Propagation) E-step.

    Parameters
    ----------
    X : np.ndarray
        Input data of shape (N, D).
    mu : np.ndarray
        Current estimate of mu of shape (D, K).
    sigma : float
        Current estimate of sigma (scalar).
    pi : np.ndarray
        Current estimate of pi of shape (1, K) or (K,) or (N, K),
        depending on usage.
    messages : np.ndarray
        Current messages array, shape (K, K, N).
    iterations : int
        Number of loopy belief propagation iterations.

    Returns
    -------
    current_lambda : np.ndarray
        Updated lambda of shape (N, K).
    new_messages : np.ndarray
        Updated messages of shape (K, K, N).
    """
    N = X.shape[0]
    K = pi.shape[1] if pi.ndim == 2 else pi.shape[0]  # adapt if pi is (K,) vs (1,K)

    # Update f_i(s_i) (its natural parameter is b_i)
    f = np.zeros((N, K))
    for i in range(N):
        # np.log(pi/(1-pi)) = logit(pi)
        # (X[i,:] @ mu) is shape (K,) if mu is (D,K) and X[i,:] is (D,)
        # diag(mu.T @ mu) is shape (K,) if mu is (D,K)
        f[i, :] = (
            np.squeeze(np.log(np.divide(pi, (1 - pi))))  # logit(pi)
            + (X[i, :] @ mu)
            - np.divide(np.diag(mu.T @ mu), 2 * sigma**2)
        )

    # Now do message passing updates for g (approximation g_tilde)
    threshold = 1e-15
    new_messages = messages.copy()

    for it in range(iterations):
        prev_messages = new_messages.copy()
        for datapoint in range(N):
            for i in range(K):
                for j in range(i + 1, K):
                    # Message from i to j
                    W = -(mu[:, j].T @ mu[:, i]) / (sigma**2)
                    other_terms = (
                        f[datapoint, i]
                        + np.sum(new_messages[:, i, datapoint])
                        - new_messages[j, i, datapoint]
                    )
                    new_messages[i, j, datapoint] = 0.5 * np.log(
                        (1 + np.exp(other_terms + W))
                        / (1 + np.exp(other_terms))
                    ) + 0.5 * new_messages[i, j, datapoint]

                    # Message from j to i
                    W = -(mu[:, i].T @ mu[:, j]) / (sigma**2)
                    other_terms = (
                        f[datapoint, i]
                        + np.sum(new_messages[:, j, datapoint])
                        - new_messages[i, j, datapoint]
                    )
                    new_messages[j, i, datapoint] = (
                        np.log(
                            (1 + np.exp(other_terms + W))
                            / (1 + np.exp(other_terms))
                        )
                        + 0.5 * new_messages[j, i, datapoint]
                    )

        # Compute current_lambda
        current_lambda = np.zeros((N, K))
        for datapoint in range(N):
            # sum over messages from all j != i
            sum_messages = np.sum(new_messages[:, :, datapoint], axis=1)
            current_lambda[datapoint, :] = 1.0 / (
                1 + np.exp(-f[datapoint, :] - sum_messages)
            )

        # Ensure numerical stability
        current_lambda[current_lambda <= 0] = 1e-15
        current_lambda[current_lambda >= 1] = 1 - 1e-15

        # Check if messages have converged
        if np.max(np.abs(prev_messages - new_messages)) < threshold:
            return current_lambda, new_messages

    return current_lambda, new_messages


def loopy_BP(X, K, iterations):
    """
    Runs loopy belief propagation by alternating between
    the EP E-step and the (mean-field) M-step.

    Parameters
    ----------
    X : np.ndarray
        Input data of shape (N, D).
    K : int
        Number of features / latent dimensions.
    iterations : int
        Number of overall iterations (E-step + M-step).

    Returns
    -------
    mu : np.ndarray
        Learned mu of shape (D, K).
    sigma : float
        Learned sigma (scalar).
    pi : np.ndarray
        Learned pi of shape (1, K).
    F_list : list
        Free energy values over iterations.
    """
    N = X.shape[0]
    F_list = []
    F_old = -np.inf
    threshold = 1e-20

    # Initialize parameters
    mu, sigma, pi, lambda_init, init_messages = initial_params(X, K)
    lambda_old = lambda_init.copy()
    messages = init_messages.copy()

    for i in range(iterations):
        # E-step
        lambda_new, new_messages = EP_E_step(X, mu, sigma, pi, messages, 50)
        lambda_new[lambda_new <= 0] = 1e-15
        lambda_new[lambda_new >= 1] = 1 - 1e-15

        # M-step (same as in Q3)
        ES = lambda_new
        ESS = np.zeros((K, K))
        for n in range(N):
            matrix = np.outer(lambda_new[n, :], lambda_new[n, :])
            # Correct the diagonal to just be lambda_new[n,:] instead of squares.
            # However, the code sets diagonal = lambda_new[n,:],
            # effectively "not" counting the product lambda_i * lambda_i?
            # That might be the intended approach for an Ising or Bernoulli model.
            np.fill_diagonal(matrix, lambda_new[n, :])
            ESS += matrix

        mu, sigma, pi = mStep(X, ES, ESS)

        # Update lambda
        lambda_old = lambda_new

        # Calculate new free energy after M-step (for debugging/checking).
        # Not fully meaningful under EP, but we track it anyway.

        # Check if F has converged
        messages = new_messages

    return mu, sigma, pi, F_list


def initial_params(X, K):
    """
    Initialize parameters mu, sigma, pi, lambda, and messages.

    Parameters
    ----------
    X : np.ndarray
        Input data of shape (N, D).
    K : int
        Number of features / latent dimensions.

    Returns
    -------
    mu : np.ndarray
        Initialized mu of shape (D, K).
    sigma : float
        Initialized sigma (scalar).
    pi : np.ndarray
        Initialized pi of shape (1, K).
    lambda_init : np.ndarray
        Initialized lambda of shape (N, K).
    init_messages : np.ndarray
        Initialized messages of shape (K, K, N).
    """
    N, D = X.shape
    lambda_init = np.random.rand(N, K)
    lambda_init[lambda_init <= 0] = 1e-15

    # Calculate initial expectations
    ES = lambda_init
    ESS = np.zeros((K, K))
    for n in range(N):
        matrix = np.outer(lambda_init[n, :], lambda_init[n, :])
        np.fill_diagonal(matrix, lambda_init[n, :])
        ESS += matrix

    mu, sigma, pi = m_step(X, ES, ESS)

    # Initialize messages
    init_messages = np.random.rand(K, K, N)
    for datapoint in range(N):
        np.fill_diagonal(init_messages[:, :, datapoint], 0.0)

    return mu, sigma, pi, lambda_init, init_messages


if __name__ == "__main__":
    # Example usage
    K = 8
    iterations = 100

    # Load data
    Y = np.loadtxt('data_images.txt')  # Adjust path as needed

    # Run loopy belief propagation
    mu, sigma, pi, F = loopy_BP(Y, K, iterations)

    # Plot free energy evolution
    plt.figure()
    plt.plot(range(1, len(F) + 1), F)
    plt.xlabel("Iteration")
    plt.ylabel("Free Energy")
    plt.title("Free Energy over Iterations")
    plt.savefig("6_free_energy.pdf", format="pdf", bbox_inches="tight")
    plt.show()

    # Plot features
    fig, axs = plt.subplots(1, K, figsize=(2 * K, 2))
    fig.suptitle(f'Features for K = {K}', fontsize=16)

    for i in range(K):
        axs[i].imshow(np.reshape(mu[:, i], (4, 4)), cmap='gray')
        axs[i].axis('off')

    plt.savefig("6_features.pdf", format="pdf", bbox_inches="tight")
    plt.show()
