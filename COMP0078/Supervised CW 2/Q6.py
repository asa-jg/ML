import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

class PolyKernel:
    def __init__(self, num_classes, kernel=None, degree=3):
        """
        Initialization
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

        # Train a separate binary classifier for each class
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


# Function to split data into train and test sets
def train_test(data, test_ratio=0.2):
    np.random.shuffle(data)
    split_index = int(len(data) * (1 - test_ratio))
    train_data = data[:split_index]
    test_data = data[split_index:]
    return train_data, test_data


# Worker function for a single run
def run_single_iteration(run, train_X, train_y, test_X, test_y, degree, epochs):
    print(f"run = {run}")
    # Initialize and train the PolyKernel model
    poly_kernel = PolyKernel(num_classes=10, degree=degree)
    poly_kernel.train(train_X, train_y, epochs=epochs)

    # Test the model
    y_pred = poly_kernel.predict(test_X)

    # Identify misclassified indices
    misclassified_indices = np.where(y_pred != test_y)[0]
    return misclassified_indices


# Load dataset
combo = np.loadtxt('zipcombo.dat', usecols=range(257))

# Split the data into train and test sets
train_data, test_data = train_test(combo)
train_X, train_y = train_data[:, 1:], train_data[:, 0].astype(int)
test_X, test_y = test_data[:, 1:], test_data[:, 0].astype(int)

# Initialize misclassification counter
misclassification_counter = np.zeros(len(test_y), dtype=int)

# Perform 50 runs using ThreadPoolExecutor
runs = 75
degree = 3
epochs = 12
max_workers = 8  # Adjust this to match the number of CPU cores

with ThreadPoolExecutor(max_workers=max_workers) as executor:
    # Map runs to threads
    futures = [executor.submit(run_single_iteration, run, train_X, train_y, test_X, test_y, degree, epochs)
               for run in range(runs)]

    # Collect results and update misclassification counter
    for future in futures:
        misclassified_indices = future.result()
        misclassification_counter[misclassified_indices] += 1

# Identify all items misclassified 50 or more times
frequently_misclassified_indices = np.where(misclassification_counter >= runs)[0]
frequently_misclassified_counts = misclassification_counter[frequently_misclassified_indices]
frequently_misclassified_images = test_X[frequently_misclassified_indices]
frequently_misclassified_labels = test_y[frequently_misclassified_indices]

# Plot all digits misclassified 50 or more times
num_items = len(frequently_misclassified_indices)
plt.figure(figsize=(15, 3 * ((num_items + 4) // 5)))  # Dynamically adjust figure height
for i, index in enumerate(frequently_misclassified_indices):
    image = frequently_misclassified_images[i].reshape((16, 16))
    plt.subplot((num_items + 4) // 5, 5, i + 1)
    plt.imshow(image, cmap='gray')
    plt.title(f"Label: {frequently_misclassified_labels[i]}\nMisclassified: {frequently_misclassified_counts[i]} times")
    plt.axis('off')
plt.tight_layout()
plt.show()
