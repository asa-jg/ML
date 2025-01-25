import numpy as np
import matplotlib.pyplot as plt

## LATEX INITIALISATION ##
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')


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


m_lst = [100, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000]
mean_opt_k = []

for m in m_lst:
    opt_k_vals = []
    for _ in range(100):
        centres, labels_h = hypothesis()
        X_train = []
        y_train = []
        for _ in range(m):
            x_i, y_i = sample(centres, labels_h)
            X_train.append(x_i)
            y_train.append(y_i)
        X_train = np.array(X_train)
        y_train = np.array(y_train)

        X_test = []
        y_test = []
        for _ in range(1000):
            x_i, y_i = sample(centres, labels_h)
            X_test.append(x_i)
            y_test.append(y_i)
        X_test = np.array(X_test)
        y_test = np.array(y_test)

        min_error = float('inf')
        ck = 1
        for k in range(1, 50):
            y_pred = knn(X_train, y_train, X_test, k)
            error = np.mean(y_pred != y_test)
            if error < min_error:
                min_error = error
                ck = k

        opt_k_vals.append(ck)

    mean_k = np.mean(opt_k_vals)
    mean_opt_k.append(mean_k)
    print(f"m = {m}, Mean Optimal k = {mean_k:.2f}")


## PLOTTING ##
plt.figure(figsize=(10, 6))
plt.plot(m_lst, mean_opt_k, marker='o')
plt.xlabel(r'\textbf{Number of Training Points} $m$', fontsize=14)
plt.ylabel(r'\textbf{Mean Optimal} $k$', fontsize=14)
plt.title(r'\textbf{Optimal} $k$ \textbf{vs. Training Set Size} $m$', fontsize=16)
plt.grid(True)
plt.show()
