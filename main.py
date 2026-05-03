import numpy as np
from face_detector import load_dataset
from pca_recognizer import PCAFaceRecognizer
from metrics import accuracy_score, compute_roc, plot_roc


def train_test_split_custom(X, y, test_ratio=0.3):
    # Custom train-test split function (No scikit-learn needed)
    np.random.seed(42)
    indices = np.random.permutation(len(X))
    test_size = int(len(X) * test_ratio)

    test_idx, train_idx = indices[:test_size], indices[test_size:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def main():
    print("1 & 2. Loading dataset and detecting faces...")
    dataset_path = "data"  # Path to your dataset folder
    X, y, class_names = load_dataset(dataset_path)

    if len(X) == 0:
        print("Error: No images found. Please check the 'data' folder structure.")
        return

    print(f"Successfully loaded {len(X)} images for {len(class_names)} classes.")

    # Split dataset into Training and Testing sets
    X_train, X_test, y_train, y_test = train_test_split_custom(X, y, test_ratio=0.3)

    print("\n3. Training PCA (Eigenfaces) From Scratch...")
    # Adjust n_components based on how many total images you have in training
    n_components = min(40, len(X_train) - 1)

    recognizer = PCAFaceRecognizer(n_components=n_components)
    recognizer.fit(X_train, y_train)

    print("Recognizing test faces...")
    predictions, distances = recognizer.predict(X_test)

    print("\n4. Reporting Performance...")
    acc = accuracy_score(y_test, predictions)
    print(f"Overall Accuracy: {acc * 100:.2f}%")

    # Plot ROC curve for the first person in the dataset (One-vs-Rest approach)
    target_class_id = 0
    if len(class_names) > 0:
        print(f"Plotting ROC curve for Class: {class_names[target_class_id]}")
        fpr, tpr = compute_roc(y_test, distances, pos_label=target_class_id)
        plot_roc(fpr, tpr, class_names[target_class_id])


if __name__ == "__main__":
    main()