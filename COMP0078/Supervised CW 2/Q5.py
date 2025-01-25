import numpy as np
from collections import defaultdict
import time


start = time.time()
#
# np.random.seed(111)

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


def compute_confusion_matrix_with_std(data, labels, num_classes=10, runs=20, test_size=0.2, epochs=5, degrees=range(1, 4), k_folds=5):
    """
    Computes the averaged confusion matrix and its standard deviation over multiple runs using cross-validation.
    """
    def split_data(data, labels, test_size, seed):
        np.random.seed(seed)
        indices = np.random.permutation(len(data))
        split_idx = int(len(data) * (1 - test_size))
        return (data[indices[:split_idx]], data[indices[split_idx:]],
                labels[indices[:split_idx]], labels[indices[split_idx:]])

    confusion_matrices = []  # To store confusion matrices for each run
    class_counts_list = []  # To store counts for normalization for each run

    for run in range(runs):
        print(f"Run {run + 1}/{runs}...")  # Progress feedback

        # Split the data for this run
        X_train, X_test, y_train, y_test = split_data(data, labels, test_size, seed=run)

        # Use cross-validation to find the best degree
        best_degree = crossval(X_train, y_train, num_classes, epochs, degrees, k_folds)

        # Train the classifier using the best degree
        model = PolyKernel(num_classes=num_classes, degree=best_degree)
        model.train(X_train, y_train, epochs)

        # Make predictions on the test set
        y_pred = model.predict(X_test)

        # Initialize confusion matrix and class counts for this run
        confusion_matrix = np.zeros((num_classes, num_classes))
        class_counts = np.zeros(num_classes)

        # Update the confusion matrix for this run
        for true, pred in zip(y_test, y_pred):
            if true != pred:  # Only record errors
                confusion_matrix[true, pred] += 1
            class_counts[true] += 1

        # Store the confusion matrix and class counts for this run
        confusion_matrices.append(confusion_matrix)
        class_counts_list.append(class_counts)

    # Normalize all confusion matrices by their respective class counts
    normalized_confusion_matrices = []
    for cm, cc in zip(confusion_matrices, class_counts_list):
        normalized_cm = cm / cc[:, None]  # Normalize by class counts
        normalized_confusion_matrices.append(normalized_cm)

    # Compute mean and standard deviation of the confusion matrices
    mean_conf_matrix = np.mean(normalized_confusion_matrices, axis=0)
    std_conf_matrix = np.std(normalized_confusion_matrices, axis=0)

    return mean_conf_matrix, std_conf_matrix


# Example Usage
# Load the data (replace with your actual dataset loading code)
bigData = np.loadtxt("zipcombo.dat")
labels = bigData[:, 0].astype(int)
features = bigData[:, 1:]

# Compute the confusion matrix and standard deviation
mean_conf_matrix, std_conf_matrix = compute_confusion_matrix_with_std(
    features, labels, num_classes=10, runs=20   , epochs=5, degrees=range(1, 4), k_folds=5
)

# Print the mean confusion matrix
np.set_printoptions(precision=4, suppress=True)
print("Mean Confusion Matrix (Error Rates):")
print(mean_conf_matrix)

# Print the standard deviation confusion matrix
print("Standard Deviation Confusion Matrix:")
print(std_conf_matrix)