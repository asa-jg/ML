import numpy as np
import time

start = time.time()

## General functions / classes  ##

np.random.seed(111)

class PolyKernel:
    def __init__(self, num_classes, kernel=None, degree=3):
        """
        Initialisation
        """
        self.num_classes = num_classes
        self.degree = degree
        self.kernel = kernel or self.poly
        self.alphas = []
        self.data = None
        self.kernel_matrix = None

    def poly(self, X, Y):
        """
        Computes the polynomial kernel between two matrices X and Y.
        """
        return (np.dot(X, Y.T) + 1) ** self.degree

    def train(self, X, y, epochs=5):
        """
        Trains the Kernel
        """
        n_samples = X.shape[0]
        self.data = X

        # Precompute kernel matrix
        self.kernel_matrix = self.kernel(X, X)

        ## Train a separate binary classifier for each class ##
        for c in range(self.num_classes):
            alpha = np.zeros(n_samples)
            b = np.where(y == c, 1, -1)

            for epoch in range(epochs):
                # Shuffle data
                indices = np.arange(n_samples)
                np.random.shuffle(indices)

                for i in indices:
                    xi_kernel_row = self.kernel_matrix[i, :]
                    pred = np.sum(alpha * xi_kernel_row)

                    # Mistake-driven update
                    if b[i] * pred <= 0:
                        alpha[i] += b[i]

            self.alphas.append(alpha)

    def predict(self, X):
        """
        Computes prediction
        """
        n_samples = X.shape[0]
        scores = np.zeros((n_samples, self.num_classes))

        # Compute confidence scores for each class
        for c in range(self.num_classes):
            # Kernel matrix: Compute K(self.data, X)
            kernel_matrix = self.kernel(self.data, X)

            # Weighted sum of kernel matrix by alpha
            scores[:, c] = np.sum(self.alphas[c][:, None] * kernel_matrix, axis=0)

        # Assign the class with the highest score
        return np.argmax(scores, axis=1)

    def test(self, X, y):
        """
        Evaluates the classifier on a test set.
        Returns the error rate.
        """
        y_pred = self.predict(X)
        error_rate = np.mean(y_pred != y)
        return error_rate


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

## Question 3 ##
def q3_runs(data, labels, num_classes=10, epochs=5):
    res = []

    for d in range(1, 6):
        trainErrors = []
        testErrors = []

        for run in range(20):
            print(f"d = {d}, run = {run}")
            X_train, X_test, y_train, y_test = split(data, labels, test_size=0.2, seed=run)

            perc = PolyKernel(num_classes=num_classes, degree=d)
            perc.train(X_train, y_train, epochs=epochs)

            train_error = perc.test(X_train, y_train)
            test_error = perc.test(X_test, y_test)

            trainErrors.append(train_error)
            testErrors.append(test_error)
        print(f"d = {d} finished")
        train_mean, train_std = np.mean(trainErrors), np.std(trainErrors)
        test_mean, test_std = np.mean(testErrors), np.std(testErrors)

        res.append([train_mean, train_std, test_mean, test_std])

    return res


bigData = np.loadtxt("zipcombo.dat")
labels = bigData[:, 0].astype(int)
features = bigData[:, 1:]

# Q3 ##
results = q3_runs(features, labels, num_classes=10, epochs=5)
print("Degree | Train Error Mean ± Std | Test Error Mean ± Std")
for d, result in enumerate(results, start=1):
    print(f"{d} | {result[0]:.4f} ± {result[1]:.4f} | {result[2]:.4f} ± {result[3]:.4f}")

end = time.time()
print(f"Execution Time: {end - start} seconds")
