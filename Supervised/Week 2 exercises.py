import numpy as np
import matplotlib.pyplot as plt

## PARAMETERS ##
d = 1
n = 100
noise_level = 0.1
random_seed = 54
degree = 10
max_n = 1000
degree_range = [1, 3, 5, 7, 9]  # Different polynomial degrees to test

## FUNCTIONS ##
def generate_linear_dataset(d=1, n=100, noise_level=0.1, random_seed=random_seed):
    '''Generate a dataset following a linear model in R^d with parametric noise level'''
    np.random.seed(random_seed)
    X = np.random.rand(n, d)
    w_true = np.random.randn(d)
    noise = noise_level * np.random.randn(n)
    y = X @ w_true + noise
    return X, y, w_true

def polynomial_feature_map(X, degree):
    '''Generate polynomial features for input data X up to a given degree'''
    X_poly = np.hstack([X**p for p in range(degree + 1)])
    return X_poly

def fit_polynomial_least_squares(X, y, degree):
    '''Fit a polynomial model using least squares for a specified degree'''
    X_poly = polynomial_feature_map(X, degree)
    X_pseudo_inverse = np.linalg.pinv(X_poly)
    w_hat = X_pseudo_inverse @ y
    return w_hat, X_poly

def polynomial_kernel(X1, X2, degree):
    '''Compute the polynomial kernel matrix between two sets of input features'''
    return (1 + X1 @ X2.T) ** degree

def fit_kernelized_polynomial_least_squares(X, y, degree):
    '''Fit a polynomial model in the dual space using the polynomial kernel'''
    K = polynomial_kernel(X, X, degree)  # Kernel matrix
    alpha = np.linalg.pinv(K) @ y  # Solve for dual coefficients
    return alpha, K

def predict_kernelized_polynomial(X_train, X_test, alpha, degree):
    '''Make predictions for the kernelized polynomial model'''
    K_test = polynomial_kernel(X_test, X_train, degree)
    y_pred = K_test @ alpha
    return y_pred

def plot_polynomial_data_and_fit(X, y, w_hat, degree):
    '''Plot the data and fitted polynomial model for a given degree (Primal)'''
    plt.figure(figsize=(8, 6))
    plt.scatter(X, y, label='Data')
    x_vals = np.linspace(0, 1, 100).reshape(-1, 1)
    x_vals_poly = polynomial_feature_map(x_vals, degree)
    y_vals = x_vals_poly @ w_hat
    plt.plot(x_vals, y_vals, color='red', label=f'Polynomial Fit (degree={degree})')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title(f'Polynomial Least Squares Fit for Degree {degree}')
    plt.legend()
    plt.show()

def plot_kernelized_data_and_fit(X, y, alpha, degree):
    '''Plot the data and fitted kernelized polynomial model for a given degree (Dual)'''
    plt.figure(figsize=(8, 6))
    plt.scatter(X, y, label='Data')
    x_vals = np.linspace(0, 1, 100).reshape(-1, 1)
    y_pred_kernelized = predict_kernelized_polynomial(X, x_vals, alpha, degree)
    plt.plot(x_vals, y_pred_kernelized, color='green', label=f'Kernelized Polynomial Fit (degree={degree})')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title(f'Kernelized Polynomial Least Squares Fit for Degree {degree}')
    plt.legend()
    plt.show()

def plot_error_vs_n_for_degrees(d=5, max_n=1000, step=100, noise_level=0.1, degrees=[1, 3, 5, 7, 9]):
    '''Plot mean squared error as a function of the number of points n for different polynomial degrees'''
    ns = list(range(20, max_n + 1, step))

    plt.figure(figsize=(10, 7))

    for degree in degrees:
        errors = []
        for n in ns:
            X, y, _ = generate_linear_dataset(d=d, n=n, noise_level=noise_level)
            alpha, _ = fit_kernelized_polynomial_least_squares(X, y, degree)
            y_pred = predict_kernelized_polynomial(X, X, alpha, degree)
            errors.append(np.mean((y - y_pred) ** 2))

        plt.plot(ns, errors, marker='o', label=f'Degree {degree}')

    plt.xlabel('Number of Points (n)')
    plt.ylabel('Mean Squared Error')
    plt.title(f'Error as a Function of n for d = {d} and Various Polynomial Degrees')
    plt.legend()
    plt.grid(True)
    plt.show()

## EXECUTION ##
plot_error_vs_n_for_degrees(d=d, max_n=max_n, noise_level=noise_level, degrees=degree_range)

## EXECUTION ##
# Example with degree p = 3
X, y, w_true = generate_linear_dataset(d=d, n=n, noise_level=noise_level)

# Primal Fit
w_hat, X_poly = fit_polynomial_least_squares(X, y, degree)
plot_polynomial_data_and_fit(X, y, w_hat, degree=degree)

# Dual Fit
alpha, K = fit_kernelized_polynomial_least_squares(X, y, degree)
plot_kernelized_data_and_fit(X, y, alpha, degree=degree)


