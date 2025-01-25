import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

# Enable LaTeX rendering in matplotlib
rc("text", usetex=True)
rc("font", family="serif")

def load(file_path):
    """Load CO2 data from the file and extract relevant columns."""
    data = np.genfromtxt(file_path, skip_header=70, usecols=(2, 3))
    years, co2_levels = data[:, 0], data[:, 1]
    return years, co2_levels

def periodic_term(s, t, tau, sigma):
    """First component of the kernel function."""
    return np.exp(-2 * (np.sin(np.pi * (s - t) / tau) ** 2) / sigma ** 2)

def rbf_term(s, t, eta):
    """Second component of the kernel function."""
    return np.exp(-((s - t) ** 2) / (2 * eta ** 2))

def kernel(X1, X2, params):
    """Construct the kernel matrix for inputs X1 and X2 using the kernel function."""
    K = np.zeros((len(X1), len(X2)))
    for i, x1 in enumerate(X1):
        for j, x2 in enumerate(X2):
            periodic = periodic_term(x1, x2, params['tau'], params['sigma'])
            rbf = rbf_term(x1, x2, params['eta'])
            noise = params['zeta'] ** 2 if x1 == x2 else 0
            K[i, j] = params['theta'] ** 2 * (periodic + params['phi'] ** 2 * rbf) + noise
    return K

def gaussian_process_prediction(X_train, Y_train, X_pred, residuals, params):
    """Perform Gaussian Process prediction."""
    K_train = kernel(X_train, X_train, params)
    K_pred_train = kernel(X_pred, X_train, params)
    K_pred_pred = kernel(X_pred, X_pred, params)

    K_train_inv = np.linalg.pinv(K_train)

    mean = K_pred_train @ K_train_inv @ residuals
    covariance = K_pred_pred - K_pred_train @ K_train_inv @ K_pred_train.T
    std_dev = np.sqrt(np.maximum(np.diag(covariance), 0))  # Avoid negative variances due to numerical errors

    return mean, std_dev

def extrapolate():
    """Main function to extrapolate CO2 concentrations using GP."""
    file_path = "co2.txt"
    X, Y = load(file_path)

    ## MAP parameters from (a) ##
    a_map, b_map = 1.8184, -3266.2

    residuals = Y - (a_map * X + b_map)
    kernel_params = {
        'theta': 2.5,
        'tau': 1,
        'sigma': 1.5,
        'phi': 0.75,
        'eta': 1.25,
        'zeta': 2   
    }

    prediction_years = np.linspace(2020, 2040, 12 * 20)

    mean, std_dev = gaussian_process_prediction(X, Y, prediction_years, residuals, kernel_params)

    f_pred = a_map * prediction_years + b_map + mean

    plt.figure(figsize=(10, 6), dpi= 200)
    plt.plot(X, Y, label=r"Observed CO$_2$", color="black")
    plt.plot(prediction_years, f_pred, color="blue", label=r"Predictive Mean $f(t)$")
    plt.fill_between(prediction_years, f_pred - std_dev, f_pred + std_dev, color="blue", alpha=0.3, label=r"$\pm 1$ Std Dev")
    plt.title(r"Extrapolated CO$_2$ Concentrations", fontsize=16)
    plt.xlabel(r"Year", fontsize=14)
    plt.ylabel(r"CO$_2$ (ppm)", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("extrapolated_co2.pdf", format="pdf")
    plt.show()

if __name__ == "__main__":
    extrapolate()
