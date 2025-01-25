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

class PolyKernelOvO:
    def __init__(self, num_classes, degree=3):
        """
        Initialisation for One-Versus-One PolyKernel.
        """
        self.num_classes = num_classes
        self.degree = degree
        self.kernel = self.poly
        self.classifiers = []  # Store binary classifiers for each pair of classes

    def poly(self, X, Y):
        """
        Computes the polynomial kernel between two matrices X and Y.
        """
        return (np.dot(X, Y.T) + 1) ** self.degree

    def train(self, X, y, epochs=5):
        """
        Trains the Kernel for One-Versus-One classification.
        """
        classes = np.unique(y)
        for i, class1 in enumerate(classes):
            for j, class2 in enumerate(classes):
                if i < j:
                    # Extract binary problem for classes class1 and class2
                    idx = (y == class1) | (y == class2)
                    X_binary = X[idx]
                    y_binary = np.where(y[idx] == class1, 1, -1)

                    # Train binary classifier
                    alpha = np.zeros(len(y_binary))
                    kernel_matrix = self.kernel(X_binary, X_binary)

                    for epoch in range(epochs):
                        indices = np.random.permutation(len(y_binary))
                        for k in indices:
                            pred = np.dot(alpha, kernel_matrix[k, :])
                            if y_binary[k] * pred <= 0:
                                alpha[k] += y_binary[k]

                    self.classifiers.append((alpha, X_binary, class1, class2))

    def predict(self, X):
        """
        Predicts class labels using majority voting.
        """
        votes = np.zeros((X.shape[0], self.num_classes))

        for alpha, X_train, class1, class2 in self.classifiers:
            kernel_matrix = self.kernel(X_train, X)
            preds = np.dot(alpha, kernel_matrix)
            votes[:, class1] += (preds > 0).astype(int)
            votes[:, class2] += (preds <= 0).astype(int)

        return np.argmax(votes, axis=1)

    def test(self, X, y):
        """
        Evaluates the classifier on a test set.
        Returns the error rate.
        """
        y_pred = self.predict(X)
        error_rate = np.mean(y_pred != y)
        return error_rate

def crossval_ovo(X, y, num_classes, epochs, degrees, k_folds=5):
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

            model = PolyKernelOvO(num_classes=num_classes, degree=d)
            model.train(X_train, y_train, epochs)
            error = model.test(X_val, y_val)
            errors.append(error)

        avg_error = np.mean(errors)
        if avg_error < min_error:
            min_error = avg_error
            best_degree = d

    return best_degree

def q8_ovo_q4(data, labels, num_classes=10, epochs=5):
    degrees = range(1, 8)
    train_res = []
    test_res = []
    best_d_values = []

    for run in range(20):
        print(f"Run = {run + 1}")
        X_train, X_test, y_train, y_test = split(data, labels, test_size=0.2, seed=run)
        best_d = crossval_ovo(X_train, y_train, num_classes, epochs, degrees)
        best_d_values.append(best_d)

        model = PolyKernelOvO(num_classes=num_classes, degree=best_d)
        model.train(X_train, y_train, epochs)

        train_error = model.test(X_train, y_train)
        test_error = model.test(X_test, y_test)

        train_res.append(train_error)
        test_res.append(test_error)

    train_mean, train_std = np.mean(train_res), np.std(train_res)
    test_mean, test_std = np.mean(test_res), np.std(test_res)
    best_d_mean = np.mean(best_d_values)

    print(f"Best Degree Mean: {best_d_mean:.2f}")
    print(f"Train Error: {train_mean:.4f} ± {train_std:.4f}")
    print(f"Test Error: {test_mean:.4f} ± {test_std:.4f}")

    return train_mean, train_std, test_mean, test_std, best_d_mean

# Example Usage
bigData = np.loadtxt("zipcombo.dat")
labels = bigData[:, 0].astype(int)
features = bigData[:, 1:]

# Run One-Versus-One Q4 protocol
q8_ovo_q4(features, labels, num_classes=10, epochs=5)
