import numpy as np
from concurrent.futures import ThreadPoolExecutor


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

    if c_values is None:
        c_values = [0.05, 0.1, 1, 5, 10]

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

    def q3_gaussian(c_values):
        res = []

        for c in c_values:
            trainErrors = []
            testErrors = []

            for run in range(20):
                print(f"run = {run+1}")
                X_train, X_test, y_train, y_test = split(data, labels, test_size=0.2, seed=run)
                model = GaussianKernel(num_classes=num_classes, c=c)
                model.train(X_train, y_train, epochs)

                train_error = model.test(X_train, y_train)
                test_error = model.test(X_test, y_test)

                trainErrors.append(train_error)
                testErrors.append(test_error)

            train_mean, train_std = np.mean(trainErrors), np.std(trainErrors)
            test_mean, test_std = np.mean(testErrors), np.std(testErrors)

            res.append([c, train_mean, train_std, test_mean, test_std])

        return res

    q3_results = q3_gaussian(c_values)

    print("C | Train Error Mean ± Std | Test Error Mean ± Std")
    for result in q3_results:
        print(f"{result[0]:.2f} | {result[1]:.4f} ± {result[2]:.4f} | {result[3]:.4f} ± {result[4]:.4f}")

    return q3_results

# Example Usage
bigData = np.loadtxt("zipcombo.dat")
labels = bigData[:, 0].astype(int)
features = bigData[:, 1:]

c_values = [0.01 * i for i in range(1,11)]
print(c_values)
q7_gaussian(features, labels, num_classes=10, epochs=5, c_values=c_values)
