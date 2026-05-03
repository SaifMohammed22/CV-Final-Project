import streamlit as st
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
    # Setup the UI Layout
    st.set_page_config(page_title="Face Recognition PCA", layout="centered")
    st.title("🧑‍💻 Face Recognition System")
    st.write("### Implemented from Scratch using PCA (Eigenfaces)")
    st.markdown("---")

    dataset_path = "data"  # Path to your dataset folder

    # Create a button to start the process
    if st.button("Start Training & Evaluation", type="primary"):

        # 1. Load Dataset
        with st.spinner("Loading dataset and detecting faces..."):
            X, y, class_names = load_dataset(dataset_path)

        if len(X) == 0:
            st.error("Error: No images found. Please check the 'data' folder structure.")
            return

        st.success(f"Successfully loaded **{len(X)}** images for **{len(class_names)}** classes.")

        # 2. Split Dataset
        X_train, X_test, y_train, y_test = train_test_split_custom(X, y, test_ratio=0.3)

        # 3. Train Model
        with st.spinner("Training PCA Model From Scratch..."):
            n_components = min(40, len(X_train) - 1)
            recognizer = PCAFaceRecognizer(n_components=n_components)
            recognizer.fit(X_train, y_train)

        # 4. Predict
        with st.spinner("Testing Model on unseen data..."):
            predictions, distances = recognizer.predict(X_test)

        st.markdown("---")

        # 5. Display Performance Metrics
        st.subheader("📊 Performance Metrics")
        acc = accuracy_score(y_test, predictions)

        # Display accuracy beautifully using Streamlit metrics
        st.metric(label="Overall Accuracy", value=f"{acc * 100:.2f}%")

        # 6. Display ROC Curve
        st.subheader("📈 ROC Curve")
        target_class_id = 0
        if len(class_names) > 0:
            st.write(f"Showing ROC curve for Class: **{class_names[target_class_id]}** (One-vs-Rest)")

            # Compute False Positive Rate and True Positive Rate
            fpr, tpr = compute_roc(y_test, distances, pos_label=target_class_id)

            # Get the figure from metrics.py and display it in Streamlit
            fig = plot_roc(fpr, tpr, class_names[target_class_id])
            st.pyplot(fig)


if __name__ == "__main__":
    main()