import numpy as np
import matplotlib.pyplot as plt


def periodic_kernel(s, t, theta=1.0, tau=1.0, sigma=1.0, phi=1.0, eta=1.0, zeta=1e-6):
    """
    Compute the kernel value k(s, t) using the given equation.

    Parameters:
    - s, t: Input points.
    - theta, tau, sigma, phi, eta, zeta: Hyperparameters.

    Returns:
    - Kernel value k(s, t).
    """
    periodic_part = np.exp(-2 * (np.sin(np.pi * (s - t) / tau) ** 2) / sigma ** 2)
    rbf_part = phi ** 2 * np.exp(-((s - t) ** 2) / (2 * eta ** 2))
    noise_part = zeta ** 2 if np.isclose(s, t) else 0
    return theta ** 2 * (periodic_part + rbf_part) + noise_part


def gp_sample_with_kernel(x, kernel_function, **kernel_params):
    """
    Generate samples drawn from a Gaussian Process (GP) using a given kernel function.

    Parameters:
    - x: Array of input points.
    - kernel_function: Kernel function.
    - kernel_params: Parameters for the kernel function.

    Returns:
    - f: A sample drawn from the GP evaluated at the input points x.
    """
    n = len(x)
    cov_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cov_matrix[i, j] = kernel_function(x[i], x[j], **kernel_params)

    cov_matrix += 1e-6 * np.eye(n)

    f = np.random.multivariate_normal(mean=np.zeros(n), cov=cov_matrix)
    return f


x = np.linspace(1980, 2020, 500)

kernel_params = {
    'theta': 2.5    ,
    'tau': 1.0,
    'sigma': 1.5,
    'phi': 0.75,
    'zeta': 0.01,
    'eta': 1.5
}

for i in range(100):
    f_sample = gp_sample_with_kernel(x, periodic_kernel, **kernel_params)

    plt.figure(figsize=(10, 6))
    plt.plot(x, f_sample, label='GP Sample', linewidth = 1)
    plt.title("Sample from GP with Periodic Kernel")
    plt.xlabel("Input (x)")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid()
    plt.show()
