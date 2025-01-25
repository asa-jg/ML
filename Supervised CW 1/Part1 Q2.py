import numpy as np
import matplotlib.pyplot as plt

## LATEX INITIALISATION ##
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')

## DATA ##
np.random.seed(111)
x_samples = np.random.uniform(0, 1, 30)


## FUNCTIONS ##
def g_sigma(x, sigma=0.07):
    """
    Adds noise to sin function
    """
    noise = np.random.normal(0, sigma, size=x.shape)
    return np.sin(2 * np.pi * x) ** 2 + noise


def plot_sin2(sigma=0.07, num_samples=30, plot_points=500):
    """
    Plots sin^2(2pix) with noisy data points
    """

    np.random.seed(111)
    x_samples = np.random.uniform(0, 1, num_samples)
    y_samples = g_sigma(x_samples, sigma)

    x_cont = np.linspace(0, 1, plot_points)
    y_cont = np.sin(2 * np.pi * x_cont) ** 2

    plt.figure(figsize=(8, 6), dpi=200)
    plt.plot(x_cont, y_cont, label=r'$\sin^2(2 \pi x)$', color='blue', linewidth=2)
    plt.scatter(x_samples, y_samples, color='black', label=r'\textbf{Noisy data points}', s=60, marker='x')

    plt.title(r'\textbf{Plot of} $\sin^2(2 \pi x)$ \textbf{with Noisy Data Points}', fontsize=18)
    plt.xlabel(r'\textbf{$x$}', fontsize=16)
    plt.ylabel(r'\textbf{$y$}', fontsize=16)
    plt.legend(fontsize=6)
    plt.grid(True)
    plt.show()


def backsub(R, b):
    """
    Solves upper triangular Rw = b via back substitution
    """
    n = R.shape[1]
    w = np.zeros(n)
    for i in range(n - 1, -1, -1):
        w[i] = (b[i] - np.dot(R[i, i + 1:], w[i + 1:])) / R[i, i]
    return w


def fit_poly(x, y, degree):
    """
    Fits a polynomial using QR decomposition
    """
    X = np.column_stack([x ** i for i in range(degree)])
    Q, R = np.linalg.qr(X)
    b = Q.T @ y
    return backsub(R, b)


def eval_poly(w, x):
    """
    Evaluates a polynomial
    """
    return sum(w[i] * x ** i for i in range(len(w)))


def mse(y_true, y_pred):
    """
    Computes MSE
    """
    return np.mean((y_true - y_pred) ** 2)


def plot_mvd(x, y, max_degree=18):
    """
    Plots log of MSE vs polynomial degree.
    """
    degrees = np.arange(1, max_degree + 1)
    mse_vals = [mse(y, eval_poly(fit_poly(x, y, d), x)) for d in degrees]

    plt.figure(figsize=(8, 6), dpi=200)
    plt.plot(degrees, np.log(mse_vals), marker='o', linestyle='-', color='blue', label=r'\textbf{Training Error (ln MSE)}')

    ## LATEX FORMATTING ##
    plt.title(r'\textbf{Log of MSE vs Polynomial Degree}', fontsize=16)
    plt.xlabel(r'\textbf{Polynomial Degree $k$}', fontsize=14)
    plt.ylabel(r'\textbf{ln(MSE)}', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.show()


def plot_tvd(x_train, y_train, x_test, y_test, max_degree=18):
    """
    Plots log of test MSE against polynomial degree.
    """
    degrees = np.arange(1, max_degree + 1)
    mse_values = [mse(y_test, eval_poly(fit_poly(x_train, y_train, d), x_test)) for d in degrees]

    plt.figure(figsize=(8, 6), dpi=200)
    plt.plot(degrees, np.log(mse_values), marker='o', linestyle='-', color='green', label=r'\textbf{Test Error (ln MSE)}')

    ## LATEX FORMATTING ##
    plt.title(r'\textbf{Log of Test MSE vs Polynomial Degree}', fontsize=16)
    plt.xlabel(r'\textbf{Polynomial Degree $k$}', fontsize=14)
    plt.ylabel(r'\textbf{ln(MSE)}', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.show()


def plot_poly(x, y, degrees):
    """
    Plots noisy data points and polynomial fits.
    """
    x_vals = np.linspace(0, 1, 500)

    plt.figure(figsize=(8, 6), dpi=200)
    plt.scatter(x, y, color='black', label=r'\textbf{Data points}', s=20)

    for d in degrees:
        w = fit_poly(x, y, d)
        y_vals = eval_poly(w, x_vals)
        plt.plot(x_vals, y_vals, label=rf'\textbf{{Polynomial fit (k={d})}}')

    ## LATEX FORMATTING ##
    plt.title(r'\textbf{Polynomial Fits for Degrees}', fontsize=16)
    plt.xlabel(r'\textbf{$x$}', fontsize=14)
    plt.ylabel(r'\textbf{$y$}', fontsize=14)
    plt.legend(fontsize=8)
    plt.grid(True)
    plt.show()


def avg_mse(runs, max_degree=18, num_train=30, num_test=1000):
    """
    Computes average training and testing MSE over multiple runs.
    """
    degrees = np.arange(1, max_degree + 1)
    train_errors = np.zeros((runs, len(degrees)))
    test_errors = np.zeros((runs, len(degrees)))

    x_test = np.random.uniform(0, 1, num_test)
    y_test = g_sigma(x_test)

    for run in range(runs):
        np.random.seed(run)
        x_train = np.random.uniform(0, 1, num_train)
        y_train = g_sigma(x_train)

        for i, d in enumerate(degrees):
            w = fit_poly(x_train, y_train, d)
            train_errors[run, i] = mse(y_train, eval_poly(w, x_train))
            test_errors[run, i] = mse(y_test, eval_poly(w, x_test))

    avg_train = np.mean(train_errors, axis=0)
    avg_test = np.mean(test_errors, axis=0)

    return degrees, avg_train, avg_test


def plot_avg(degrees, avg_train_errors, avg_test_errors):
    """
    Plots log of average training and testing MSE over multiple runs.
    """
    plt.figure(figsize=(8, 6), dpi=200)

    plt.plot(degrees, np.log(avg_train_errors), marker='o', linestyle='-', color='blue',
             label=r'\textbf{Avg. Training Error (ln MSE)}')

    plt.plot(degrees, np.log(avg_test_errors), marker='o', linestyle='-', color='red',
             label=r'\textbf{Avg. Test Error (ln MSE)}')

    ## LATEX FORMATTING ##
    plt.title(r'\textbf{Average Log of Training and Test MSE over Runs}', fontsize = 16)
    plt.xlabel(r'\textbf{Polynomial Degree $k$}', fontsize=14)
    plt.ylabel(r'\textbf{ln(Avg MSE)}', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.show()



## (a) (i) ##
plot_sin2(sigma=0.07, num_samples=30, plot_points=500)


## (a) (ii) ##
y_samples = g_sigma(x_samples)
k = [2,5,10,14,18]
degrees = [kval for kval in k]
plot_poly(x_samples, y_samples, degrees)


## (b) ##
plot_mvd(x_samples, y_samples)


## (c) ##
x_test = np.random.uniform(0, 1, 1000)
y_test = g_sigma(x_test)
plot_tvd(x_samples, y_samples, x_test, y_test)

## (d) ##
runs = 100
degrees, train_errors, test_errors = avg_mse(runs, max_degree=18)
plot_avg(degrees, train_errors, test_errors)

