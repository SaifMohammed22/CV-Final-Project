import streamlit as st
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from face_detector import load_dataset, detect_and_crop_faces, EMOTION_MAPPING
from pca_recognizer import PCAFaceRecognizer
from metrics import accuracy_score, compute_roc, plot_roc

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def get_emotion_name(emotion_code):
    """Convert emotion code to full name. E.g., 'AN' -> 'Angry'"""
    return EMOTION_MAPPING.get(emotion_code, emotion_code)


def train_test_split_custom(X, y, test_ratio=0.3, image_paths=None):
    np.random.seed(42)
    indices = np.random.permutation(len(X))
    test_size = int(len(X) * test_ratio)
    test_idx, train_idx = indices[:test_size], indices[test_size:]

    if image_paths is not None:
        image_paths_array = np.array(image_paths)
        return (X[train_idx], X[test_idx], y[train_idx], y[test_idx],
                image_paths_array[train_idx], image_paths_array[test_idx])
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def compute_per_emotion_accuracy(y_true, y_pred, class_names):
    """Compute accuracy per emotion."""
    results = {}
    for idx, emotion in enumerate(class_names):
        mask = y_true == idx
        if mask.sum() > 0:
            acc = (y_pred[mask] == y_true[mask]).mean()
            results[emotion] = {
                'accuracy': acc * 100,
                'count': mask.sum()
            }
    return results


def get_sample_by_emotion(dataset_path, limit_per_emotion=1):
    """Get one sample image per emotion to show data preview."""
    samples = {}

    for root, _, files in os.walk(dataset_path):
        for image_name in sorted(files):
            if os.path.splitext(image_name)[1].lower() not in IMAGE_EXTENSIONS:
                continue

            # Extract emotion from filename (e.g., KA.AN1.39.tiff -> AN)
            parts = os.path.splitext(image_name)[0].split('.')
            if len(parts) >= 2:
                emotion = parts[1][:2]

                if emotion not in samples:
                    image_path = os.path.join(root, image_name)
                    samples[emotion] = image_path

                    if len(samples) == 7:  # All 7 emotions
                        return samples

    return samples


def main():
    st.set_page_config(page_title="JAFFE Emotion Recognition", layout="wide")

    # Compact header
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        st.title("🎭 Emotion Recognition")
        st.caption("PCA/Eigenfaces on JAFFE Dataset")

    dataset_path = "data/jaffe"

    # Show sample data preview
    with st.expander("📸 View Sample Data", expanded=True):
        samples = get_sample_by_emotion(dataset_path)
        if samples:
            cols = st.columns(7)
            for idx, (emotion, image_path) in enumerate(sorted(samples.items())):
                with cols[idx]:
                    img = cv2.imread(image_path)
                    if img is not None:
                        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        st.image(img_gray, caption=f"{emotion}\n{get_emotion_name(emotion)}",
                                 use_container_width=True)
        st.caption(
            "One sample per emotion from the JAFFE dataset (213 images total)")

    st.divider()

    # Quick start button
    if st.button("▶ Train & Evaluate", type="primary", use_container_width=True):

        # Load and process
        with st.spinner("Loading JAFFE dataset..."):
            X, y, class_names, image_paths = load_dataset(dataset_path)

        if len(X) == 0:
            st.error("❌ No images found in data/jaffe")
            return

        # Display dataset stats in compact format
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Images", len(X))
        with col2:
            st.metric("Emotions", len(class_names))
        with col3:
            st.metric("Posers", 10)

        st.divider()

        # Split and train
        X_train, X_test, y_train, y_test, train_paths, test_paths = train_test_split_custom(
            X, y, test_ratio=0.3, image_paths=image_paths)

        with st.spinner("Training PCA model..."):
            n_components = min(40, len(X_train) - 1)
            recognizer = PCAFaceRecognizer(n_components=n_components)
            recognizer.fit(X_train, y_train)

        with st.spinner("Running predictions..."):
            predictions, distances = recognizer.predict(X_test)

        # Main results - use tabs
        tab1, tab2, tab3 = st.tabs(["📊 Summary", "📈 Analysis", "🖼️ Gallery"])

        # TAB 1: Summary
        with tab1:
            overall_acc = accuracy_score(y_test, predictions)
            per_emotion = compute_per_emotion_accuracy(
                y_test, predictions, class_names)

            # Overall metric - large
            st.metric("Overall Accuracy", f"{overall_acc * 100:.1f}%",
                      delta=f"{len(y_test)} test samples")

            # Per-emotion breakdown - compact table
            st.subheader("Per-Emotion Performance")
            emotion_data = []
            for emotion_code, stats in sorted(per_emotion.items()):
                emotion_data.append({
                    "Emotion": f"{emotion_code} ({get_emotion_name(emotion_code)})",
                    "Accuracy": f"{stats['accuracy']:.1f}%",
                    "Samples": stats['count']
                })

            st.dataframe(emotion_data, use_container_width=True,
                         hide_index=True)

        # TAB 2: ROC & Analysis
        with tab2:
            col1, col2 = st.columns([0.55, 0.45])

            with col1:
                st.subheader("ROC Curve")
                target_class_id = 0
                if len(class_names) > 0:
                    emotion_code = class_names[target_class_id]
                    emotion_name = get_emotion_name(emotion_code)
                    st.caption(f"{emotion_code} ({emotion_name}) vs Rest")

                    fpr, tpr = compute_roc(
                        y_test, distances, pos_label=target_class_id)
                    fig = plot_roc(fpr, tpr, emotion_code)
                    st.pyplot(fig, use_container_width=True)

            with col2:
                st.subheader("Confusion Distribution")
                # Show correct vs incorrect breakdown
                correct = (predictions == y_test).sum()
                incorrect = len(y_test) - correct

                fig, ax = plt.subplots(figsize=(8, 6))
                colors = ['#2ecc71', '#e74c3c']
                wedges, texts, autotexts = ax.pie(
                    [correct, incorrect],
                    labels=['✓ Correct', '✗ Incorrect'],
                    autopct='%1.1f%%',
                    colors=colors,
                    startangle=90
                )
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')
                st.pyplot(fig, use_container_width=True)

        # TAB 3: Test Gallery
        with tab3:
            st.subheader("Sample Predictions (first 8)")
            display_limit = min(8, len(test_paths))
            cols = st.columns(4)

            for idx in range(display_limit):
                test_image_path = test_paths[idx]
                true_label_idx = y_test[idx]
                pred_label_idx = predictions[idx]

                true_emotion_code = class_names[true_label_idx]
                pred_emotion_code = class_names[pred_label_idx]
                is_correct = (true_label_idx == pred_label_idx)

                with cols[idx % 4]:
                    img = cv2.imread(test_image_path)
                    if img is not None:
                        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        st.image(img_gray, use_container_width=True)

                    # Compact result display
                    border_color = "🟢" if is_correct else "🔴"
                    st.markdown(f"**{border_color}** T: {true_emotion_code} | P: {pred_emotion_code}",
                                help=f"True: {get_emotion_name(true_emotion_code)}\nPred: {get_emotion_name(pred_emotion_code)}")


if __name__ == "__main__":
    main()
