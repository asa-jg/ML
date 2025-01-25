import numpy as np
import pandas as pd


## LOADING ##
np.random.seed(111)
df = pd.read_csv('Boston-filtered.csv')
X, y = df.iloc[:, :12].values, df.iloc[:, -1].values
runs = 20
train_errors = []
test_errors = []


## FUNCTIONS ##
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
    Calculates MSE
    """
    return np.mean((y_true - y_pred) ** 2)


def naive():
    """
    Implements a naive regression model
    """
    for run in range(runs):
        X_train, y_train, X_test, y_test = split_data(X, y)
        y_train_mean = np.mean(y_train)

        y_train_pred = np.full_like(y_train, y_train_mean)
        y_test_pred = np.full_like(y_test, y_train_mean)

        train_mse = mse(y_train, y_train_pred)
        test_mse = mse(y_test, y_test_pred)

        train_errors.append(train_mse)
        test_errors.append(test_mse)

    avg_train = np.mean(train_errors)
    avg_test = np.mean(test_errors)

    return avg_train, avg_test


def predict(X, w):
    """
    Generates predictions using a linear regression model.
    """
    X = np.c_[np.ones(X.shape[0]), X]
    return X @ w


def single_attr_lr():
    """
    Performs linear regression using a single feature and bias term for each feature.
    """
    def helper(X_train, y_train):
        X_train = np.c_[np.ones(X_train.shape[0]), X_train]
        return np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train

    for index in range(X.shape[1]):
        train_errors = []
        test_errors = []
        print(f"\nLinear Regression with single attribute: {df.columns[index]}")

        for run in range(runs):
            X_train, y_train, X_test, y_test = split_data(X[:, index].reshape(-1, 1), y)
            w = helper(X_train, y_train)

            y_train_pred = predict(X_train, w)
            y_test_pred = predict(X_test, w)

            train_errors.append(mse(y_train, y_train_pred))
            test_errors.append(mse(y_test, y_test_pred))

        avg_train= np.mean(train_errors)
        avg_test = np.mean(test_errors)

        print(f"Average Training MSE: {avg_train:.2f}")
        print(f"Average Test MSE: {avg_test:.2f}")


def all_attr_lr():
    """
    Performs linear regression using all features and a bias term.
    """
    def helper(X_train, y_train):
        X_train = np.c_[np.ones(X_train.shape[0]), X_train]
        return np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train

    train_errors = []
    test_errors = []

    for run in range(runs):
        X_train, y_train, X_test, y_test = split_data(X, y)
        w = helper(X_train, y_train)

        y_train_pred = predict(X_train, w)
        y_test_pred = predict(X_test, w)

        train_errors.append(mse(y_train, y_train_pred))
        test_errors.append(mse(y_test, y_test_pred))

    avg_train = np.mean(train_errors)
    avg_test = np.mean(test_errors)

    print(f"\nAverage Training MSE with all attributes: {avg_train:.2f}")
    print(f"Average Test MSE with all attributes: {avg_test:.2f}")


## (a) ##
train_e, test_e = naive()
print(f"\nAverage Training MSE: {train_e:.2f}")
print(f"Average Test MSE: {test_e:.2f}")

## (c) ##
single_attr_lr()

## (d) ##
all_attr_lr()
