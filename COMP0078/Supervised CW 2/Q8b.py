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
        self.pairs = []  # Store pairs of classes

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

def q8_ovo(data, labels, num_classes=10, epochs=5):
    results = []

    for d in range(1, 8):
        trainErrors = []
        testErrors = []

        for run in range(20):
            print(f"Degree = {d}, Run = {run + 1}")
            X_train, X_test, y_train, y_test = split(data, labels, test_size=0.2, seed=run)

            model = PolyKernelOvO(num_classes=num_classes, degree=d)
            model.train(X_train, y_train, epochs=epochs)

            train_error = model.test(X_train, y_train)
            test_error = model.test(X_test, y_test)

            trainErrors.append(train_error)
            testErrors.append(test_error)

        train_mean, train_std = np.mean(trainErrors), np.std(trainErrors)
        test_mean, test_std = np.mean(testErrors), np.std(testErrors)

        results.append([d, train_mean, train_std, test_mean, test_std])

    print("Degree | Train Error Mean ± Std | Test Error Mean ± Std")
    for result in results:
        print(f"{result[0]} | {result[1] * 100:.2f}% ± {result[2] * 100:.2f}% | {result[3] * 100:.2f}% ± {result[4] * 100:.2f}%")

    return results

# Example Usage
bigData = np.loadtxt("zipcombo.dat")
labels = bigData[:, 0].astype(int)
features = bigData[:, 1:]

# Run One-Versus-One protocol
q8_ovo(features, labels, num_classes=10, epochs=5)
