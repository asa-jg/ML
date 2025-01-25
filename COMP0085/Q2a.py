import numpy as np
import pandas as pd

def q2a(file_path):
    ## Load data ##
    data = pd.read_csv(file_path, delim_whitespace=True, header=None, names=["Year", "Month", "DecimalYear", "CO2", "Junk"])
    yearVals, co2Vals = data["DecimalYear"].values, data["CO2"].values

    ## Priors ##
    mean_prior = np.array([0, 360])
    prec_prior = np.diag([1 / 100, 1 / 10000])

    ## Posteriors via lecture slides ##
    TTM = np.array([[np.sum(yearVals**2), yearVals.sum()], [yearVals.sum(), len(yearVals)]])
    precMatrix = prec_prior + TTM
    covariancePosterior = np.linalg.inv(precMatrix)
    y_transpose = np.column_stack((yearVals, np.ones_like(yearVals))).T @ co2Vals
    meanPosterior = covariancePosterior @ (prec_prior @ mean_prior + y_transpose)

    return meanPosterior, covariancePosterior

if __name__ == "__main__":
    file_path = "co2.txt"
    posterior_mean, posterior_cov = q2a(file_path)
    print("Posterior Mean (a, b):", posterior_mean)
    print("Posterior Covariance:\n", posterior_cov)



