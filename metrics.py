import numpy as np
import matplotlib.pyplot as plt


def accuracy_score(y_true, y_pred):
    # Calculate the percentage of correct predictions
    correct_predictions = np.sum(y_true == y_pred)
    total_predictions = len(y_true)
    return correct_predictions / total_predictions


def compute_roc(y_true, y_distances, pos_label):
    # Convert distances to similarity scores (Negative distance means higher is closer/better)
    scores = -y_distances

    # Binarize labels for One-vs-Rest ROC curve
    y_binary = (y_true == pos_label).astype(int)

    # Get unique thresholds sorted descending
    thresholds = np.sort(np.unique(scores))[::-1]

    tpr_list = []
    fpr_list = []

    # Total actual positives and negatives
    P = np.sum(y_binary)
    N = len(y_binary) - P

    # Calculate True Positive Rate and False Positive Rate for each threshold
    for thresh in thresholds:
        y_pred = (scores >= thresh).astype(int)

        TP = np.sum((y_pred == 1) & (y_binary == 1))
        FP = np.sum((y_pred == 1) & (y_binary == 0))

        TPR = TP / P if P > 0 else 0
        FPR = FP / N if N > 0 else 0

        tpr_list.append(TPR)
        fpr_list.append(FPR)

    return np.array(fpr_list), np.array(tpr_list)


def plot_roc(fpr, tpr, class_name):
    # Create a figure object to be passed to the Streamlit UI
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve for {class_name}')
    ax.legend(loc="lower right")
    ax.grid(True)

    # Return the figure instead of showing it directly
    return fig
