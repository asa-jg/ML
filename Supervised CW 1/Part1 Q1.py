import numpy as np
import matplotlib.pyplot as plt

## LATEX INITIALISING ##
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')

## DATA ##
x = np.array([1, 2, 3, 4])
y = np.array([3, 2, 0, 5])
degrees = [1, 2, 3, 4]


## FUNCTIONS ##
def fit(x, y, degree):
    """Fits a polynomial"""
    X = np.column_stack([x ** i for i in range(degree)])
    w = np.linalg.inv(X.T @ X) @ X.T @ y
    return w


def evaluate(w, x_vals):
    """Evaluates a polynomial"""
    return sum(w[i] * x_vals ** i for i in range(len(w)))


def plot_poly(x, y, degrees):
    """Plots polynomials"""
    x_vals = np.linspace(0.5, 4.5, 100)

    plt.figure(figsize=(8, 6), dpi=200)
    plt.scatter(x, y, color='black', label=r'\textbf{Data points}')

    for degree in degrees:
        w = fit(x, y, degree)
        y_vals = evaluate(w, x_vals)
        label = f"{['Constant', 'Linear', 'Quadratic', 'Cubic'][degree - 1]} fit ($k={degree}$)"
        plt.plot(x_vals, y_vals, label=rf'\textbf{{{label}}}')

    ## LATEX FORMATTING ##
    plt.title(r'\textbf{Polynomial Fits for Constant, Linear, Quadratic, and Cubic}', fontsize=16)
    plt.xlabel(r'\textbf{x}', fontsize=14)
    plt.ylabel(r'\textbf{y}', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.show()


def print_polyeq(x, y, degrees):
    """Prints the polynomial equations"""

    def poly(w):
        terms = [f"{w[i]:.2f}x^{i}" if i > 0 else f"{w[i]:.2f}" for i in range(len(w))]
        equation = " + ".join(terms).replace("x^1", "x")
        return equation

    for degree in degrees:
        w = fit(x, y, degree)
        label = f"{['Constant', 'Linear', 'Quadratic', 'Cubic'][degree -1]}:"
        print(f"Equation for {label} y = {poly(w)}")


def print_mse(x, y, degrees):
    """Calculates and prints the MSE"""
    def mse(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    for degree in degrees:
        w = fit(x, y, degree)
        y_pred = evaluate(w, x)
        mse_val = mse(y, y_pred)
        label = f"{['Constant', 'Linear', 'Quadratic', 'Cubic'][degree -1]}"
        print(f"MSE for {label} fit ($k={degree}$): {mse_val:.4f}")



## (a) ##
plot_poly(x, y, degrees)

## (b) ##
print_polyeq(x, y, degrees)

## (c)##
print_mse(x, y, degrees)
