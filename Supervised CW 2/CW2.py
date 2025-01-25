import numpy as np
import matplotlib.pyplot as plt
import time

start = time.time()



## General functions / classes / methods ##

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


class GaussKernel:
    def __init__(self, num_classes, kernel=None, c=1.0):
        """
        Initialisation
        :param num_classes: Number of classes for OvA.
        :param kernel: Custom kernel function; default is Gaussian kernel.
        :param c: Width parameter for the Gaussian kernel.
        """
        self.num_classes = num_classes
        self.c = c
        self.kernel = kernel or self.gaussian
        self.alphas = []
        self.data = None

    def gaussian(self, X, Y):
        """
        Computes the Gaussian kernel between two matrices X and Y.
        :param X: Matrix of shape (n_samples_X, n_features).
        :param Y: Matrix of shape (n_samples_Y, n_features).
        :return: Kernel matrix of shape (n_samples_X, n_samples_Y).
        """
        X_norm = np.sum(X ** 2, axis=1).reshape(-1, 1)
        Y_norm = np.sum(Y ** 2, axis=1).reshape(1, -1)
        squared_distances = X_norm + Y_norm - 2 * np.dot(X, Y.T)
        return np.exp(-self.c * squared_distances)

    def train(self, X, y, epochs=5):
        """
        Trains the Kernel
        :param X: Training data, shape (n_samples, n_features).
        :param y: Labels, shape (n_samples,).
        :param epochs: Number of epochs.
        """
        n_samples = X.shape[0]
        self.data = X

        ## Train a separate binary classifier for each class ##
        for c in range(self.num_classes):
            alpha = np.zeros(n_samples)
            b = np.where(y == c, 1, -1)

            for epoch in range(epochs):
                for i in range(n_samples):
                    xi = X[i]

                    # Compute prediction
                    pred = np.sum(alpha * self.kernel(X, xi.reshape(1, -1)).flatten())

                    # Update if misclassified
                    if b[i] * pred <= 0:
                        alpha[i] += b[i]

            self.alphas.append(alpha)

    def predict(self, X):
        """
        Computes predictions for test data.
        :param X: Test data, shape (n_samples, n_features).
        :return: Predicted labels, shape (n_samples,).
        """
        n_samples = X.shape[0]
        scores = np.zeros((n_samples, self.num_classes))

        # Compute confidence scores for each class
        for c in range(self.num_classes):
            # Kernel matrix: Compute K(self.data, X) -> shape (n_train, n_test)
            kernel_matrix = self.kernel(self.data, X)

            # Weighted sum of kernel matrix by alpha: shape (n_test,)
            scores[:, c] = np.sum(self.alphas[c][:, None] * kernel_matrix, axis=0)

        # Assign the class with the highest score
        return np.argmax(scores, axis=1)

    def test(self, X, y):
        """
        Evaluates the classifier on a test set.
        :param X: Test data, shape (n_samples, n_features).
        :param y: True labels, shape (n_samples,).
        :return: Accuracy score.
        """
        y_pred = self.predict(X)
        accuracy = np.mean(y_pred == y)
        return accuracy




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

    train= indices[:split]
    test = indices[split:]

    X_train, X_test = data[train], data[test]
    y_train, y_test = labels[train], labels[test]

    return X_train, X_test, y_train, y_test

## Question 3 ##
def q3_runs(data, labels, num_classes=10, epochs=5):
    res = []

    for d in range(1, 8):
        trainErrors = []
        testErrors = []

        for run in range(20):
            X_train, X_test, y_train, y_test = split(data, labels, test_size=0.2, seed=run)

            perc = PolyKernel(num_classes=num_classes, degree=d)
            perc.train(X_train, y_train, epochs=epochs)

            trainSingle = perc.test(X_train, y_train)
            testSingle = perc.test(X_test, y_test)

            trainErrors.append(trainSingle)
            testErrors.append(testSingle)
        print(f"d = {d} finished")
        train_mean, train_std = np.mean(trainErrors), np.std(trainErrors)
        test_mean, test_std = np.mean(testErrors), np.std(testErrors)

        res.append([train_mean, train_std, test_mean, test_std])

    return res

def conf(y_true, y_pred, num_classes):
    """
    Computes a confusion matrix for mcc
    """
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(y_true, y_pred):
        matrix[true, pred] += 1
    return matrix


def q4_q5(data, labels, num_classes=10, epochs=5):
    degrees = range(1, 8)
    train_res = []
    test_res = []
    conf_lst = np.zeros((num_classes, num_classes))  # Confusion matrix aggregated over runs

    for run in range(20):
        print(f"run = {run}")
        X_train, X_test, y_train, y_test = split(data, labels, test_size=0.2, seed=run)

        # Precompute kernel matrix once for training data
        kernel_matrix_train = np.dot(X_train, X_train.T) + 1
        kernel_matrix_test = np.dot(X_test, X_train.T) + 1

        best_d, _ = crossval(X_train, y_train, num_classes, epochs, degrees, kernel_matrix_train)

        perc = PolyKernel(num_classes=num_classes, degree=best_d)
        perc.train(X_train, y_train, epochs=epochs)

        train_error = 1 - perc.test(X_train, y_train)
        test_error = 1 - perc.test(X_test, y_test)

        train_res.append(train_error)
        test_res.append(test_error)

        y_pred = perc.predict(X_test)
        conf_matrix = conf(y_test, y_pred, num_classes)
        conf_lst += conf_matrix

    norm = conf_lst / 20
    errors = norm / np.sum(norm, axis=1, keepdims=True)

    train_mean, train_std = np.mean(train_res), np.std(train_res)
    test_mean, test_std = np.mean(test_res), np.std(test_res)

    return train_mean, train_std, test_mean, test_std, errors

def crossval(X, y, num_classes, epochs, degrees, kernel_matrix_train, k_folds=5):
    """
    Performs cross validation with precomputed kernel matrix
    """
    fold_size = len(X) // k_folds
    validation_errors = {d: [] for d in degrees}

    for d in degrees:
        for fold in range(k_folds):
            val_start = fold * fold_size
            val_end = (fold + 1) * fold_size

            X_val = X[val_start:val_end]
            y_val = y[val_start:val_end]

            train_indices = list(range(0, val_start)) + list(range(val_end, len(X)))

            kernel_train = kernel_matrix_train[np.ix_(train_indices, train_indices)] ** d
            kernel_val = kernel_matrix_train[np.ix_(range(val_start, val_end), train_indices)] ** d

            X_train = X[train_indices]
            y_train = y[train_indices]

            # Train on training folds
            perceptron = PolyKernel(num_classes=num_classes, degree=d)
            perceptron.train(X_train, y_train, epochs=epochs)

            # Validate on validation fold
            val_error = 1 - perceptron.test(X_val, y_val)
            validation_errors[d].append(val_error)

    # Average validation error for each degree
    avg_val_errors = {d: np.mean(validation_errors[d]) for d in degrees}
    best_d = min(avg_val_errors, key=avg_val_errors.get)  # Degree with lowest error

    return best_d, avg_val_errors[best_d]



# def q6(features, labels, num_classes, epochs, runs=20):
#     """
#     Identifies the five hardest-to-predict samples based on misclassification frequency.
#     :param features: Feature matrix, shape (n_samples, n_features).
#     :param labels: True labels, shape (n_samples,).
#     :param num_classes: Number of classes.
#     :param epochs: Number of epochs for training.
#     :param runs: Number of runs for tracking misclassifications.
#     :return: List of indices of the hardest-to-predict samples.
#     """
#     n_samples = features.shape[0]
#     misclassification_counts = np.zeros(n_samples, dtype=int)
#
#     for run in range(runs):
#         # Split into train and test sets
#         X_train, X_test, y_train, y_test = split(features, labels, test_size=0.2, seed=run)
#
#         # Train the perceptron with a fixed degree (e.g., d=3)
#         perceptron = PolyKernel(num_classes=num_classes, degree=3)
#         perceptron.train(X_train, y_train, epochs=epochs)
#
#         # Predict on the test set
#         y_pred = perceptron.predict(X_test)
#
#         # Record misclassified samples
#         for i, (true_label, pred_label) in enumerate(zip(y_test, y_pred)):
#             if true_label != pred_label:
#                 misclassification_counts[i] += 1
#
#     # Find the indices of the five most misclassified samples
#     hardest_indices = np.argsort(misclassification_counts)[-5:]
#     return hardest_indices
#
#
# def visualise(features, labels, indices, image_shape=(16, 16)):
#     """
#     Visualises the hardest-to-predict samples.
#
#     """
#     plt.figure(figsize=(10, 5))
#     for i, idx in enumerate(indices):
#         plt.subplot(1, 5, i + 1)
#         plt.imshow(features[idx].reshape(image_shape), cmap="gray")
#         plt.title(f"Label: {labels[idx]}")
#         plt.axis("off")
#     plt.show()


bigData = np.loadtxt("zipcombo.dat")
labels = bigData[:, 0].astype(int)
features = bigData[:, 1:]


# # Q3 ##
# results = q3_runs(bigData, labels, num_classes=10, epochs=5)
# print("Degree | Train Mean ± Std | Test Mean ± Std")
# for d, result in enumerate(results, start=1):
#     print(f"{d} | {result[0]:.4f} ± {result[1]:.4f} | {result[2]:.4f} ± {result[3]:.4f}")

# Q4 and Q5 ##
train_mean, train_std, test_mean, test_std, errors = q4_q5(features, labels, num_classes=10, epochs=5)

print(f"Train Error: {train_mean:.4f} ± {train_std:.4f}")
print(f"Test Error: {test_mean:.4f} ± {test_std:.4f}")

# Output confusion matrix
print("Confusion Matrix (Error Rates):")
print(errors)

## Q6 ##
# hardest_indices = q6(features, labels, num_classes=10, epochs=5, runs=20)
# print(f"Indices of the hardest-to-predict samples: {hardest_indices}")
# visualise(features, labels, hardest_indices)


end = time.time()
print(end - start)