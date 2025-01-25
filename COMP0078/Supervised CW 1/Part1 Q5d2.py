import numpy as np
import pandas as pd

## LOADING ##
np.random.seed(111)

df = pd.read_csv('Boston-filtered.csv')
X, y = df.iloc[:, :12].values, df.iloc[:, -1].values


def split_data(X, y, train_size=0.67):
    """
    Splits the dataset into training and test sets.
    """
    num = X.shape[0]
    train_indices = np.random.choice(num, int(train_size * num), replace=False)
    test_indices = np.setdiff1d(np.arange(num), train_indices)
    return X[train_indices], y[train_indices], X[test_indices], y[test_indices]


def mse(y_true, y_pred):
    """
    Calculates the MSE
    """
    return np.mean((y_true - y_pred) ** 2)


def kernel(X1, X2, sigma):
    """
    Computes the Gaussian kernel matrix between two datasets.
    """
    sq_dists = np.sum(X1 ** 2, axis=1).reshape(-1, 1) + np.sum(X2 ** 2, axis=1) - 2 * np.dot(X1, X2.T)
    return np.exp(-sq_dists / (2 * sigma ** 2))


def krr(K, y, gamma):
    """
    Performs Kernel Ridge Regression
    """
    n = K.shape[0]
    alpha = np.linalg.inv(K + gamma * np.eye(n)) @ y
    return alpha


def kfold(X, y, k=5):
    """
    Implements k-fold
    """
    n = X.shape[0]
    indices = np.arange(n)
    np.random.shuffle(indices)
    fold_size = n // k
    folds = [indices[i * fold_size:(i + 1) * fold_size] for i in range(k)]
    for i in range(k):
        val_indices = folds[i]
        train_indices = np.hstack([folds[j] for j in range(k) if j != i])
        yield train_indices, val_indices


def cv_krr(X_train, y_train, gamma_values, sigma_values, k=5):
    """
    Performs cross-validation for KRR
    """
    best_mse = float('inf')
    best_gamma, best_sigma = None, None
    cv = []

    for gamma in gamma_values:
        for sigma in sigma_values:
            fold_mses = []

            for train_indices, val_indices in kfold(X_train, y_train, k):
                Xf_train, Xf_val = X_train[train_indices], X_train[val_indices]
                yf_train, yf_val = y_train[train_indices], y_train[val_indices]

                K_train = kernel(Xf_train, Xf_train, sigma)
                K_val = kernel(Xf_val, Xf_train, sigma)

                alpha = krr(K_train, yf_train, gamma)
                y_pred_val = K_val @ alpha

                fold_mses.append(mse(yf_val, y_pred_val))

            avg_mse = np.mean(fold_mses)
            cv.append((gamma, sigma, avg_mse))

            if avg_mse < best_mse:
                best_mse = avg_mse
                best_gamma, best_sigma = gamma, sigma

    return best_gamma, best_sigma, best_mse, cv


gamma_values = [2 ** -i for i in range(40, 25, -1)]
sigma_values = [2 ** (7 + i / 2) for i in range(0, 13)]

runs = 20
train_lst = []
test_lst = []
param_lst = []

for run in range(runs):
    print(f"\nRun {run + 1}/{runs}")

    X_train, y_train, X_test, y_test = split_data(X, y)

    best_gamma, best_sigma, _, _ = cv_krr(X_train, y_train, gamma_values, sigma_values)
    param_lst.append((best_gamma, best_sigma))

    k_train = kernel(X_train, X_train, best_sigma)
    alpha = krr(k_train, y_train, best_gamma)

    y_train_pred = k_train @ alpha
    train_mse = mse(y_train, y_train_pred)

    K_test = kernel(X_test, X_train, best_sigma)
    y_test_pred = K_test @ alpha
    test_mse = mse(y_test, y_test_pred)

    train_lst.append(train_mse)
    test_lst.append(test_mse)

    print(f"Best Gamma: {best_gamma}, Best Sigma: {best_sigma}")
    print(f"Training MSE: {train_mse:.4f}, Test MSE: {test_mse:.4f}")

mean_train = np.mean(train_lst)
mean_test = np.mean(test_lst)
std_train = np.std(train_lst)
std_test = np.std(test_lst)

print(f"Mean Training MSE: {mean_train:.4f} ± {std_train:.4f}")
print(f"Mean Test MSE: {mean_test:.4f} ± {std_test:.4f}")
