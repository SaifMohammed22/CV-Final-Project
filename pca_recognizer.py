import numpy as np


class PCAFaceRecognizer:
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean_face = None
        self.eigenfaces = None
        self.weights = None
        self.labels = None

    def fit(self, X, y):
        # X shape: (Number of samples, Number of pixels)
        self.labels = y

        # 1. Calculate the mean face
        self.mean_face = np.mean(X, axis=0)

        # 2. Subtract the mean face from all faces (phi)
        phi = X - self.mean_face

        # 3. Calculate covariance matrix
        # Trick: Use A^T * A instead of A * A^T to save massive computation memory
        covariance_matrix = np.dot(phi, phi.T)

        # 4. Calculate eigenvalues and eigenvectors
        # _, _, v = np.linalg.svm(covariance_matrix)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        # 5. Sort eigenvectors by eigenvalues in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]

        # 6. Map the eigenvectors back to the original face dimensions
        self.eigenfaces = np.dot(phi.T, eigenvectors).T

        # 7. Normalize the eigenfaces
        for i in range(self.eigenfaces.shape[0]):
            self.eigenfaces[i] = self.eigenfaces[i] / \
                np.linalg.norm(self.eigenfaces[i])

        # Keep only the top 'n_components'
        self.eigenfaces = self.eigenfaces[:self.n_components]

        # 8. Project training faces onto the eigenfaces (Get weights)
        self.weights = np.dot(phi, self.eigenfaces.T)

    def predict(self, X_test):
        # 1. Subtract mean face from test data
        phi_test = X_test - self.mean_face

        # 2. Project test faces onto the eigenfaces
        test_weights = np.dot(phi_test, self.eigenfaces.T)

        predictions = []
        min_distances = []

        # 3. Find the closest match using Euclidean Distance
        for weight in test_weights:
            distances = np.linalg.norm(self.weights - weight, axis=1)
            best_match_idx = np.argmin(distances)

            predictions.append(self.labels[best_match_idx])
            min_distances.append(distances[best_match_idx])

        return np.array(predictions), np.array(min_distances)