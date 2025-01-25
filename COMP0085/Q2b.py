import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def compute_residuals(file_path):
    data = pd.read_csv(file_path, delim_whitespace=True, header=None, names=["Year", "Month", "DecimalYear", "CO2", "Junk"])
    yearVals, co2Vals = data["DecimalYear"].values, data["CO2"].values

    mean_prior = np.array([0, 360])
    prec_prior = np.diag([1 / 100, 1 / 10000])

    TTM = np.array([[np.sum(yearVals**2), yearVals.sum()], [yearVals.sum(), len(yearVals)]])
    precMatrix = prec_prior + TTM
    covariancePosterior = np.linalg.inv(precMatrix)
    y_transpose = np.column_stack((yearVals, np.ones_like(yearVals))).T @ co2Vals
    meanPosterior = covariancePosterior @ (prec_prior @ mean_prior + y_transpose)

    a_MAP, b_MAP = meanPosterior
    print(a_MAP, b_MAP)
    g_obs = co2Vals - (a_MAP * yearVals + b_MAP)

    return yearVals, g_obs, meanPosterior, covariancePosterior

def plot_residuals(yearVals, g_obs):
    plt.figure(figsize=(10, 6))
    plt.plot(yearVals, g_obs, color='blue', alpha=0.7, label="Residuals")
    plt.axhline(0, color='red', linestyle='--', label="Zero Residual Line")
    plt.xlabel("Year (Decimal)")
    plt.ylabel("Residuals (g_obs)")
    plt.title("Q2b")
    plt.legend()
    plt.grid()
    plt.show()

file_path = 'co2.txt'

yearVals, g_obs, posterior_mean, posterior_cov = compute_residuals(file_path)
plot_residuals(yearVals, g_obs)

