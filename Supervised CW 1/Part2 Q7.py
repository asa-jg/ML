import numpy as np
import matplotlib.pyplot as plt

## FUNCTIONS ##
def hypothesis():
    """
    Generates random hypothesis
    """
    centres = np.random.rand(100, 2)
    labels = np.random.randint(0, 2, 100)
    return centres, labels


def h_S_v(point, centres, labels, v=3):
    """
    Computes h_S_v for a given point.
    """
    distances = np.sqrt(np.sum((centres - point) ** 2, axis=1))
    indices = np.argsort(distances)[:v]
    nearest_lbls = labels[indices]
    counts = np.bincount(nearest_lbls, minlength=2)
    if counts[0] > counts[1]:
        return 0
    elif counts[1] > counts[0]:
        return 1
    else:
        return np.random.randint(0, 2)


def sample(centres, labels):
    """
    Generates a random sample point and assigns a label
    """
    x = np.random.rand(2)
    if np.random.rand() < 0.8:
        y = h_S_v(x, centres, labels)
    else:
        y = np.random.randint(0, 2)
    return x, y


def knn(x_train, y_train, x_test, k):
    """
    Predicts labels for test points
    """
    y_pred = []
    for x in x_test:
        distances = np.sqrt(np.sum((x_train - x) ** 2, axis=1))
        indices = np.argsort(distances)[:k]
        nearest_lbls = y_train[indices]
        counts = np.bincount(nearest_lbls, minlength=2)
        if counts[0] > counts[1]:
            y_hat = 0
        elif counts[1] > counts[0]:
            y_hat = 1
        else:
            y_hat = np.random.randint(0, 2)
        y_pred.append(y_hat)
    return np.array(y_pred)


k_values = range(1, 50)
mean_errors = []

for k in k_values:
    errors = []
    for _ in range(100):  
        centres, lh = hypothesis()

        X_train = []
        y_train = []
        for _ in range(4000):
            x_i, y_i = sample(centres, lh)
            X_train.append(x_i)
            y_train.append(y_i)
        X_train = np.array(X_train)
        y_train = np.array(y_train)

        X_test = []
        y_test = []
        for _ in range(1000):
            x_i, y_i = sample(centres, lh)
            X_test.append(x_i)
            y_test.append(y_i)
        X_test = np.array(X_test)
        y_test = np.array(y_test)

        y_pred = knn(X_train, y_train, X_test, k)

        error = np.mean(y_pred != y_test)
        errors.append(error)

    mean_error = np.mean(errors)
    mean_errors.append(mean_error)
    print(f"k = {k}, Mean Error = {mean_error:.4f}")

## PLOTTING ##
plt.figure(figsize=(10, 6))
plt.plot(k_values, mean_errors, marker='o')
plt.xlabel(r'$k$ (Number of Nearest Neighbours)', fontsize=14)
plt.ylabel(r'Estimated Generalization Error', fontsize=14)
plt.title(r'Generalization Error vs. $k$ for $k$-NN Algorithm', fontsize=16)
plt.grid(True)
plt.show()
