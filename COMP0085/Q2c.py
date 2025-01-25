import numpy as np


def gp_sample(x, kernel_function):
    """
    Generate samples drawn from a Gaussian Process (GP).

    Parameters:
    - x: Array of input points (n x d, where n is the number of points, and d is the dimensionality).
    - kernel_function: Function to compute the covariance kernel, k(x, x').

    Returns:
    - f: A sample drawn from the GP evaluated at the input points x.
    """
    n = len(x)
    cov_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cov_matrix[i, j] = kernel_function(x[i], x[j])

    cov_matrix += 1e-6 * np.eye(n)

    f = np.random.multivariate_normal(mean=np.zeros(n), cov=cov_matrix)
    return f


# Example kernel function (RBF/Exponential Squared Kernel)
def rbf_kernel(x1, x2, length_scale=1.0, variance=1.0):
    """
    Radial Basis Function (RBF) kernel, also known as the Gaussian kernel.

    Parameters:
    - x1, x2: Input points.
    - length_scale: Length scale of the kernel.
    - variance: Variance of the kernel.

    Returns:
    - The kernel value for x1 and x2.
    """
    return variance * np.exp(-np.linalg.norm(x1 - x2) ** 2 / (2 * length_scale ** 2))


x = np.linspace(0, 10, 100).reshape(-1, 1)  # 100 points in 1D
f_sample = gp_sample(x, lambda x1, x2: rbf_kernel(x1, x2, length_scale=1.0, variance=1.0))

import matplotlib.pyplot as plt

plt.plot(x, f_sample)
plt.title("Sample from a Gaussian Process")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.show()
