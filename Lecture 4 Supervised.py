import numpy as np
import matplotlib.pyplot as plt


## Function to generate synthetic Gaussian data with a 'no-man's land' ##
def generate_gaussian_data_with_gap(n_points, mean_1, mean_2, cov_1, cov_2, label_1, label_2, gap_threshold):
    """
    Generate a synthetic binary classification dataset with a 'no-man's land' between two Gaussian distributions.

    Args:
    - n_points (int): Number of points to generate for each class.
    - mean_1 (tuple): Mean of the first Gaussian distribution.
    - mean_2 (tuple): Mean of the second Gaussian distribution.
    - cov_1 (tuple): Covariance matrix of the first Gaussian distribution.
    - cov_2 (tuple): Covariance matrix of the second Gaussian distribution.
    - label_1 (int): Label for the first distribution.
    - label_2 (int): Label for the second distribution.
    - gap_threshold (float): Minimum distance between points of different classes to enforce the 'no-man's land'.

    Returns:
    - X (ndarray): Generated input features (2D coordinates).
    - y (ndarray): Labels for the generated points.
    """

    def is_in_gap(point, opposite_mean, threshold):
        """Check if a point is within the 'no-man's land' region."""
        distance = np.linalg.norm(point - opposite_mean)
        return distance < threshold

    X1, y1 = [], []
    X2, y2 = [], []

    while len(X1) < n_points:
        point = np.random.multivariate_normal(mean_1, cov_1)
        if not is_in_gap(point, mean_2, gap_threshold):  # Check for 'no-man's land'
            X1.append(point)
            y1.append(label_1)

    while len(X2) < n_points:
        point = np.random.multivariate_normal(mean_2, cov_2)
        if not is_in_gap(point, mean_1, gap_threshold):  # Check for 'no-man's land'
            X2.append(point)
            y2.append(label_2)

    # Stack the data
    X = np.vstack((X1, X2))
    y = np.concatenate((y1, y2))

    return np.array(X), np.array(y)


## Function to compute the least squares closed-form solution ##
def least_squares(X, y):
    """
    Solve the linear least squares problem.

    Args:
    - X (ndarray): The input feature matrix (with bias term).
    - y (ndarray): The labels (target values).

    Returns:
    - w (ndarray): The learned weights.
    """
    X_transpose_X = np.dot(X.T, X)
    X_transpose_y = np.dot(X.T, y)
    w = np.linalg.inv(X_transpose_X).dot(X_transpose_y)

    return w


## Gradient Descent for Squared Loss ##
def gradient_descent(X, y, lr=0.01, n_iters=1000):
    """
    Perform Gradient Descent to minimize the squared loss.

    Args:
    - X (ndarray): The input feature matrix (with bias term).
    - y (ndarray): The labels.
    - lr (float): Learning rate.
    - n_iters (int): Number of iterations.

    Returns:
    - w (ndarray): The optimized weight vector.
    - errors (list): List of training errors at each iteration.
    """
    N, d = X.shape
    w = np.random.randn(d)  # Initialize weights randomly
    errors = []

    for i in range(n_iters):
        predictions = np.dot(X, w)
        error = np.mean((predictions - y) ** 2)  # Compute squared loss
        errors.append(error)

        gradient = -2 * np.dot(X.T, (y - predictions)) / N  # Compute the gradient
        w -= lr * gradient  # Update the weights

    return w, errors


## Function to plot the decision boundary ##
def plot_decision_boundary(X, y, w):
    """
    Plot the decision boundary for a 2D dataset using the learned weights.

    Args:
    - X (ndarray): Input feature matrix (without bias term).
    - y (ndarray): Labels.
    - w (ndarray): Learned weight vector.
    """
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', label='Class +1')
    plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], color='red', label='Class -1')

    # Plot the decision boundary (line)
    x_vals = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    y_vals = -(w[0] * x_vals + w[2]) / w[1]  # Decision boundary: w[0] * x + w[1] * y + w[2] = 0
    plt.plot(x_vals, y_vals, 'k-', label='Decision Boundary')

    plt.title('Decision Boundary Learned by Gradient Descent')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()


## Generate synthetic data ##
np.random.seed(42)  # For reproducibility
n_points = 100
mean_1 = [0, 0]
mean_2 = [3, 3]
cov_1 = [[1, 0.2], [0.2, 1]]
cov_2 = [[1, -0.2], [-0.2, 1]]
gap_threshold = 1.5

X, y = generate_gaussian_data_with_gap(n_points, mean_1, mean_2, cov_1, cov_2, label_1=1, label_2=-1,
                                       gap_threshold=gap_threshold)
X_bias = np.hstack((X, np.ones((X.shape[0], 1))))  # Add bias term

## Perform Gradient Descent ##
learning_rate = 0.05
n_iterations = 1000

w_gd, gd_errors = gradient_descent(X_bias, y, lr=learning_rate, n_iters=n_iterations)

## Closed-form Least Squares Solution ##
w_ls = least_squares(X_bias, y)

## Compare the final weights ##
print(f"Gradient Descent Weights: {w_gd}")
print(f"Closed-form Solution Weights: {w_ls}")

## Plot the training error vs. number of iterations ##
plt.plot(gd_errors, label='Gradient Descent')
plt.title('Training Error vs. Number of Iterations (Gradient Descent)')
plt.xlabel('Iterations')
plt.ylabel('Mean Squared Error')
plt.grid(True)
plt.legend()
plt.show()

## Plot the decision boundary learned by Gradient Descent ##
plot_decision_boundary(X, y, w_gd)
