import numpy as np

def split(data, labels, test_size=0.2, seed=0):
    """
    Splits data into training and testing subsets.
    """
    np.random.seed(seed)
    data = np.array(data)
    labels = np.array(labels)

    indices = np.arange(len(data))
    np.random.shuffle(indices)

    ## Determine which index to split at ##
    split = int(len(data) * (1 - test_size))

    train = indices[:split]
    test = indices[split:]

    X_train, X_test = data[train], data[test]
    y_train, y_test = labels[train], labels[test]

    return X_train, X_test, y_train, y_test

def q7_gaussian(data, labels, num_classes=10, epochs=5, c_values=None):
    import numpy as np

    class GaussianKernel:
        def __init__(self, num_classes, c=1.0):
            self.num_classes = num_classes
            self.c = c
            self.alphas = []
            self.data = None

        def gaussian(self, X, Y):
            X_norm = np.sum(X ** 2, axis=1).reshape(-1, 1)
            Y_norm = np.sum(Y ** 2, axis=1).reshape(1, -1)
            squared_distances = X_norm + Y_norm - 2 * np.dot(X, Y.T)
            return np.exp(-self.c * squared_distances)

        def train(self, X, y, epochs=5):
            n_samples = X.shape[0]
            self.data = X
            kernel_matrix = self.gaussian(X, X)

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
            kernel_matrix = self.gaussian(self.data, X)
            scores = np.array([
                np.dot(alpha, kernel_matrix) for alpha in self.alphas
            ])
            return np.argmax(scores, axis=0)

        def test(self, X, y):
            y_pred = self.predict(X)
            return np.mean(y_pred != y)

    def crossval_gaussian(X, y, num_classes, epochs, c_values, k_folds=5):
        fold_size = len(X) // k_folds
        best_c = None
        min_error = float("inf")

        for c in c_values:
            errors = []
            for fold in range(k_folds):
                val_start = fold * fold_size
                val_end = (fold + 1) * fold_size
                X_val = X[val_start:val_end]
                y_val = y[val_start:val_end]
                X_train = np.vstack((X[:val_start], X[val_end:]))
                y_train = np.concatenate((y[:val_start], y[val_end:]))

                model = GaussianKernel(num_classes=num_classes, c=c)
                model.train(X_train, y_train, epochs)
                error = model.test(X_val, y_val)
                errors.append(error)

            avg_error = np.mean(errors)
            if avg_error < min_error:
                min_error = avg_error
                best_c = c

        return best_c

    def q4_gaussian(c_values):
        trainErrors = []
        testErrors = []
        c_star_values = []

        for run in range(20):
            print(f"Run {run + 1} started.")
            X_train, X_test, y_train, y_test = split(data, labels, test_size=0.2, seed=run)
            best_c = crossval_gaussian(X_train, y_train, num_classes, epochs, c_values)
            c_star_values.append(best_c)

            model = GaussianKernel(num_classes=num_classes, c=best_c)
            model.train(X_train, y_train, epochs)

            train_error = model.test(X_train, y_train)
            test_error = model.test(X_test, y_test)

            trainErrors.append(train_error)
            testErrors.append(test_error)

            print(f"Run {run + 1} completed. Best c: {best_c:.2f}, Train Error: {train_error:.4f}, Test Error: {test_error:.4f}")

        train_mean, train_std = np.mean(trainErrors), np.std(trainErrors)
        test_mean, test_std = np.mean(testErrors), np.std(testErrors)
        c_star_mean = np.mean(c_star_values)

        print(f"C* Mean: {c_star_mean:.2f}")
        print(f"Train Error: {train_mean:.4f} ± {train_std:.4f}")
        print(f"Test Error: {test_mean:.4f} ± {test_std:.4f}")

        return train_mean, train_std, test_mean, test_std, c_star_mean

    return q4_gaussian(c_values)

# Example Usage
bigData = np.loadtxt("zipcombo.dat")
labels = bigData[:, 0].astype(int)
features = bigData[:, 1:]

c_values = [0.01 * i for i in range(1, 11)]
q7_gaussian(features, labels, num_classes=10, epochs=5, c_values=c_values)
