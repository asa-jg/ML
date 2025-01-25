import numpy as np
import pandas as pd

## LOADING ##
np.random.seed(111)
df = pd.read_csv('Boston-filtered.csv')
X, y = df.iloc[:, :12].values, df.iloc[:, -1].values


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
    Calculates the MSE
    """
    return np.mean((y_true - y_pred) ** 2)


def predict(X, w):
    """
    Predicts target values using the learned weights.
    """
    X = np.c_[np.ones(X.shape[0]), X]
    return X @ w


def single_attr_lr(X, y):
    """
    Performs linear regression using each feature individually.
    """
    def helper(X_train, y_train):
        """
        Trains a linear regression model on a single feature.
        """
        X_train = np.c_[np.ones(X_train.shape[0]), X_train]
        w = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train
        return w

    train_all = []
    test_all = []

    for index in range(X.shape[1]):
        train_errors = []
        test_errors = []
        print(f"\nLinear Regression with single attribute: {df.columns[index]}")

        for run in range(runs):
            X_train, y_train, X_test, y_test = split_data(X[:, index].reshape(-1, 1), y)
            w = helper(X_train, y_train)
            y_train_pred = predict(X_train, w)
            y_test_pred = predict(X_test, w)

            train_mse = mse(y_train, y_train_pred)
            test_mse = mse(y_test, y_test_pred)
            train_errors.append(train_mse)
            test_errors.append(test_mse)

        avg_train = np.mean(train_errors)
        avg_test = np.mean(test_errors)
        std_train = np.std(train_errors)
        std_test = np.std(test_errors)

        train_all.append((avg_train, std_train))
        test_all.append((avg_test, std_test))

        print(f"Average Training MSE: {avg_train:.2f} ± {std_train:.2f}")
        print(f"Average Test MSE: {avg_test:.2f} ± {std_test:.2f}")

    return train_all, test_all


def all_attr_lr(X, y):
    """
    Performs linear regression using all features together.
    """
    def helper2(X_train, y_train):
        X_train = np.c_[np.ones(X_train.shape[0]), X_train]
        w = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train
        return w

    train_errors = []
    test_errors = []

    for run in range(runs):
        X_train, y_train, X_test, y_test = split_data(X, y)
        w = helper2(X_train, y_train)
        y_train_pred = predict(X_train, w)
        y_test_pred = predict(X_test, w)

        train_mse = mse(y_train, y_train_pred)
        test_mse = mse(y_test, y_test_pred)
        train_errors.append(train_mse)
        test_errors.append(test_mse)

    avg_train = np.mean(train_errors)
    avg_test = np.mean(test_errors)
    std_train = np.std(train_errors)
    std_test = np.std(test_errors)

    print(f"\nAverage Training MSE with all attributes: {avg_train:.2f} ± {std_train:.2f}")
    print(f"Average Test MSE with all attributes: {avg_test:.2f} ± {std_test:.2f}")

    return avg_train, std_train, avg_test, std_test


runs = 20

np.random.seed(111)

## PART (a) ##
train_naive, test_naive = [], []

for run in range(runs):
    X_train, y_train, X_test, y_test = split_data(X, y)
    y_train_mean = np.mean(y_train)
    y_train_pred = np.ones_like(y_train) * y_train_mean
    y_test_pred = np.ones_like(y_test) * y_train_mean
    train_naive.append(mse(y_train, y_train_pred))
    test_naive.append(mse(y_test, y_test_pred))

avg_naive_train = np.mean(train_naive)
avg_naive_test = np.mean(test_naive)
std_naive_train = np.std(train_naive)
std_naive_test = np.std(test_naive)

print(f"Naive Regression - Training MSE: {avg_naive_train:.2f} ± {std_naive_train:.2f}")
print(f"Naive Regression - Test MSE: {avg_naive_test:.2f} ± {std_naive_test:.2f}")

## PART (c) ##
print("\nRunning Single Attribute Linear Regression")
single_attr_lr(X, y)

## PART (d) ##
print("\nRunning Linear Regression with All Attributes")
all_attr_lr(X, y)
