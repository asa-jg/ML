import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor

start = time.time()

np.random.seed(111)

class PolyKernel:
    def __init__(self, num_classes, degree=3):
        self.num_classes = num_classes
        self.degree = degree
        self.alphas = []
        self.data = None

    def poly(self, X, Y):
        """
        Computes the polynomial kernel between two matrices X and Y.
        """
        return (np.dot(X, Y.T) + 1) ** self.degree

    def train(self, X, y, epochs=5):
        """
        Trains the kernel classifier.
        """
        n_samples = X.shape[0]
        self.data = X
        kernel_matrix = self.poly(X, X)

        for c in range(self.num_classes):
            alpha = np.zeros(n_samples)
            b = np.where(y == c, 1, -1)

            for epoch in range(epochs):
                indices = np.random.permutation(n_samples)
                for i in indices:
                    pred = np.dot(alpha, kernel_matrix[i, :])
                    if b[i] * pred <= 0:
                        alpha[i] += b[i]
            self.alphas.append(alpha)

    def predict(self, X):
        """
        Predicts the class labels for the given data.
        """
        kernel_matrix = self.poly(self.data, X)
        scores = np.array([
            np.dot(alpha, kernel_matrix) for alpha in self.alphas
        ])
        return np.argmax(scores, axis=0)

    def test(self, X, y):
        """
        Computes the test error.
        """
        y_pred = self.predict(X)
        return np.mean(y_pred != y)


def split(data, labels, test_size=0.2, seed=0):
    """
    Splits the data into training and testing subsets.
    """
    np.random.seed(seed)
    indices = np.random.permutation(len(data))
    split_idx = int(len(data) * (1 - test_size))
    return (data[indices[:split_idx]], data[indices[split_idx:]],
            labels[indices[:split_idx]], labels[indices[split_idx:]])



def crossval(X, y, num_classes, epochs, degrees, k_folds=5):
    """
    Performs k-fold cross-validation to determine the best polynomial degree.
    """
    fold_size = len(X) // k_folds
    best_degree = None
    min_error = float("inf")

    for d in degrees:
        errors = []
        for fold in range(k_folds):
            val_start = fold * fold_size
            val_end = (fold + 1) * fold_size
            X_val = X[val_start:val_end]
            y_val = y[val_start:val_end]
            X_train = np.vstack((X[:val_start], X[val_end:]))
            y_train = np.concatenate((y[:val_start], y[val_end:]))

            model = PolyKernel(num_classes=num_classes, degree=d)
            model.train(X_train, y_train, epochs)
            error = model.test(X_val, y_val)
            errors.append(error)

        avg_error = np.mean(errors)
        if avg_error < min_error:
            min_error = avg_error
            best_degree = d

    return best_degree


def q4(data, labels, num_classes=10, epochs=5):
    """
    Main function for training and evaluating the kernel classifier.
    """
    degrees = range(1, 4)  # Reduced degree range for efficiency
    train_res = []
    test_res = []
    conf_lst = np.zeros((num_classes, num_classes))

    def run_single_experiment(run):
        print(f"run = {run}")
        X_train, X_test, y_train, y_test = split(data, labels, test_size=0.2, seed=run)
        best_d = crossval(X_train, y_train, num_classes, epochs, degrees)
        model = PolyKernel(num_classes=num_classes, degree=best_d)
        model.train(X_train, y_train, epochs)

        train_error = 1 - model.test(X_train, y_train)
        test_error = 1 - model.test(X_test, y_test)

        return train_error, test_error

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = executor.map(run_single_experiment, range(20))

    for train_error, test_error in results:
        train_res.append(train_error)
        test_res.append(test_error)

    train_mean, train_std = np.mean(train_res), np.std(train_res)
    test_mean, test_std = np.mean(test_res), np.std(test_res)
    norm = conf_lst / 20
    errors = norm / np.sum(norm, axis=1, keepdims=True)

    return train_mean, train_std, test_mean, test_std, errors


# Load data and execute
bigData = np.loadtxt("zipcombo.dat")
labels = bigData[:, 0].astype(int)
features = bigData[:, 1:]

train_mean, train_std, test_mean, test_std, errors = q4(features, labels, num_classes=10, epochs=5)

print(f"Train Error: {train_mean:.4f} ± {train_std:.4f}")
print(f"Test Error: {test_mean:.4f} ± {test_std:.4f}")


end = time.time()
print(f"Execution Time: {end - start:.2f} seconds")
