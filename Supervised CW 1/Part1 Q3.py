import numpy as np
import matplotlib.pyplot as plt

## LATEX INITIALISATION ##
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')

## DATA GENERATION ##
np.random.seed(111)
x_samples = np.random.uniform(0, 1, 30)


## FUNCTIONS ##
def g_sigma(x, sigma=0.07):
    """
    Adds noise to sin function
    """
    noise = np.random.normal(0, sigma, size=x.shape)
    return np.sin(2 * np.pi * x) ** 2 + noise


def trig(x_samples, degree):
    """
    Creates a trigonometric basis.
    """
    return np.column_stack([np.sin((i + 1) * np.pi * x_samples) for i in range(degree)])


def backsub(R, b):
    """
    Solves upper triangular Rw = b via back substitution
    """
    n = R.shape[1]
    w = np.zeros(n)
    for i in range(n - 1, -1, -1):
        w[i] = (b[i] - np.dot(R[i, i + 1:], w[i + 1:])) / R[i, i]
    return w


def fit_trig(x_samples, y_samples, degree):
    """
    Fits using QR decomposition.
    """
    X = trig(x_samples, degree)
    Q, R = np.linalg.qr(X)
    b = Q.T @ y_samples
    return backsub(R, b)


def eval_trig(w, x_vals):
    """
    Evaluates the trig basis
    """
    return sum(w[i] * np.sin((i + 1) * np.pi * x_vals) for i in range(len(w)))


def mse(y_true, y_pred):
    """
    Computes MSE
    """
    return np.mean((y_true - y_pred) ** 2)


def plot_trigmvd(x_samples, y_samples, max_degree=18):
    """
    Plots the log of training MSE against the degree of the trigonometric basis.
    """
    degrees = np.arange(1, max_degree + 1)
    mse_values = []
    for d in degrees:
        w = fit_trig(x_samples, y_samples, d)
        y_pred = eval_trig(w, x_samples)
        mse_values.append(mse(y_samples, y_pred))

    ## LATEX FORMATTING ##
    plt.figure(figsize=(8, 6), dpi=200)
    plt.plot(degrees, np.log(mse_values), marker='o', linestyle='-', color='blue', label=r'\textbf{Training Error (ln MSE)}')
    plt.title(r'\textbf{Log of Training MSE vs Degree (Trigonometric Basis)}', fontsize=16)
    plt.xlabel(r'\textbf{Degree $k$}', fontsize=14)
    plt.ylabel(r'\textbf{ln(MSE)}', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.show()


def plot_trigtvd(x_samples, y_samples, x_test, y_test, max_degree=18):
    """
    Plots the log of test MSE against degree
    """
    degrees = np.arange(1, max_degree + 1)
    mse_values = []
    for d in degrees:
        w = fit_trig(x_samples, y_samples, d)
        y_test_pred = eval_trig(w, x_test)
        mse_values.append(mse(y_test, y_test_pred))

    ## LATEX FORMATTING ##
    plt.figure(figsize=(8, 6), dpi=200)
    plt.plot(degrees, np.log(mse_values), marker='o', linestyle='-', color='green', label=r'\textbf{Test Error (ln MSE)}')
    plt.title(r'\textbf{Log of Test MSE vs Degree (Trigonometric Basis)}', fontsize=16)
    plt.xlabel(r'\textbf{Degree $k$}', fontsize=14)
    plt.ylabel(r'\textbf{ln(MSE)}', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.show()


def average_mse_trig(runs, max_degree=18, num_training_points=30, num_test_points=1000):
    """
    Computes average training and testing MSE over multiple runs
    """
    degrees = np.arange(1, max_degree + 1)
    train_errors = np.zeros((runs, len(degrees)))
    test_errors = np.zeros((runs, len(degrees)))
    x_test = np.random.uniform(0, 1, num_test_points)
    y_test = g_sigma(x_test)

    for run in range(runs):
        np.random.seed(run)
        x_samples = np.random.uniform(0, 1, num_training_points)
        y_samples = g_sigma(x_samples)
        for i, d in enumerate(degrees):
            w = fit_trig(x_samples, y_samples, d)
            y_train_pred = eval_trig(w, x_samples)
            train_errors[run, i] = mse(y_samples, y_train_pred)
            y_test_pred = eval_trig(w, x_test)
            test_errors[run, i] = mse(y_test, y_test_pred)

    avg_train = np.mean(train_errors, axis=0)
    avg_test = np.mean(test_errors, axis=0)
    return degrees, avg_train, avg_test


def plot_avg_trig(degrees, avg_train_errors, avg_test_errors):
    """
    Plots log of average training and test MSE over multiple runs
    """
    ## LATEX FORMATTING ##
    plt.figure(figsize=(8, 6), dpi=200)
    plt.plot(degrees, np.log(avg_train_errors), marker='o', linestyle='-', color='blue', label=r'\textbf{Avg. Training Error (ln MSE)}')
    plt.plot(degrees, np.log(avg_test_errors), marker='o', linestyle='-', color='red', label=r'\textbf{Avg. Test Error (ln MSE)}')
    plt.title(r'\textbf{Avg Log of Training and Test MSE over 100 Runs (Trig Basis)}', fontsize=16)
    plt.xlabel(r'\textbf{Degree $k$}', fontsize=14)
    plt.ylabel(r'\textbf{ln(Avg MSE)}', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.show()



## (b) ##
y_samples = g_sigma(x_samples)

plot_trigmvd(x_samples, y_samples)

## (c) ##
x_test = np.random.uniform(0, 1, 1000)
y_test = g_sigma(x_test)
plot_trigtvd(x_samples, y_samples, x_test, y_test)

## (d) ##
runs = 100
degrees, avg_train, avg_test = average_mse_trig(runs, max_degree=18)
plot_avg_trig(degrees, avg_train, avg_test)
