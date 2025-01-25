import numpy as np
from matplotlib import pyplot as plt


def EMAlgorithm(K, X, iter_max, tor):
    # initialisation
    # fix part
    N, D = X.shape
    X_1 = np.ones(X.shape) - X
    r = np.zeros((N, K))
    LL = []
    # random part
    pi = np.random.random((1, K))
    pi /= sum(pi)  # normalisation
    p = np.random.random((K, D))

    # EM iteration
    for i in range(iter_max):
        p_1 = np.ones(p.shape) - p  # matrix 1-p

        # E-step / responsibility
        for n in range(N):
            r[n, :] = np.multiply(pi,
                                  np.prod(np.power(p, X[n, :]) * np.power(p_1, X_1[n, :]), axis=1))  # multiply by row
            r[n, :] /= sum(r[n, :])

        # M-step
        pi = np.sum(r, axis=0) / N  # sum by column
        # np.testing.assert_allclose(pi.sum(-1), 1.0)
        p = np.matmul(r.T, X) / np.matmul(np.sum(r, axis=0).reshape((K, 1)), np.ones((1, D)))

        # log-likelihood
        loglik = 0
        for n in range(N):
            loglik += np.log(np.sum(np.multiply(pi, np.prod(np.power(p, X[n, :]) * np.power(p_1, X_1[n, :]), axis=1))))
        LL.append(loglik)

        if len(LL) >= 2:
            if LL[-1] - LL[-2] < tor:
                break

    return pi, p, LL


def loglikplots(K, LL, indx):
    plt.figure()
    plt.plot(range(0, len(LL)), LL)
    plt.title(f"loglikelihood when k={K} with final loglik={LL[-1]}")
    plt.savefig(f"K={K}_seed={indx}.pdf")


def paraplots(K, p, pi, indx):
    if K == 10:
        fig, axes = plt.subplots(2, 5, figsize=(16, 9))
        i = 0
        axes = axes.flatten()
        for ax in axes:
            ax.imshow(np.reshape(p[i, :], (8, 8)),
                      interpolation="nearest",
                      cmap="gray")
            ax.set_title(f"weight {pi[i]:.3f}", fontsize=20)
            i += 1
        plt.savefig(f"model_size{K}_seed{indx}.pdf")

    elif K == 7:
        fig, axes = plt.subplots(2, 4, figsize=(16, 9))
        fig.delaxes(axes[-1][-1])
        i = 0
        axes = axes.flatten()
        for ax in axes:
            if i == 7:
                break
            else:
                ax.imshow(np.reshape(p[i, :], (8, 8)),
                          interpolation="nearest",
                          cmap="gray")
                ax.set_title(f"weight {pi[i]:.3f}", fontsize=20)
                i += 1
        plt.savefig(f"model_size{K}_seed{indx}.pdf")

    else:
        fig, axes = plt.subplots(1, K, figsize=(3 * K, 3))
        for i in range(K):
            ax = axes[i]
            ax.imshow(np.reshape(p[i, :], (8, 8)),
                      interpolation="nearest",
                      cmap='gray')
            ax.set_title(f"weight {pi[i]:.3f}", fontsize=20)
        plt.savefig(f"model_size{K}_seed{indx}.pdf")


if __name__ == "__main__":
    X = np.loadtxt('binarydigits.txt')

    # question(d)
    # fix initialisation for different Ks
    K = [2, 3, 4, 7, 10]
    for k in K:
        np.random.seed(1)
        # run EM
        pi, p, LL = EMAlgorithm(k, X, 500, 1e-10)
        # save loglikelihood plots
        loglikplots(k, LL, 1)
        # save parameter plots (pi and p)
        paraplots(k, p, pi, 1)

    # question(e)
    # fix K for different initialisations
    K = 2
    for i in range(5):
        np.random.seed(i)
        # run EM
        pi, p, LL = EMAlgorithm(K, X, 500, 1e-10)
        # save loglikelihood plots
        loglikplots(K, LL, i)
        # save parameter plots (pi and p)
        paraplots(K, p, pi, i)


