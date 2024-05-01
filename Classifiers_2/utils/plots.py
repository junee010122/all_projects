import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

import numpy as np

def plot_confusion_matrix(labels, preds, class_names, title, path):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(8, 8))
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    plt.title(title)
    fig.colorbar(cax)
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=0)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names)
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(path + title + '.png')  # Save the plot with the title as the filename


def plot_performance_metrics(test_labels, predicted_labels, title, path):
    """
    Plots accuracy, precision, recall, and F1-score as a bar chart.
    """
    metrics = {
        'Accuracy': accuracy_score(test_labels, predicted_labels),
        'Precision': precision_score(test_labels, predicted_labels, average='macro', zero_division=0),
        'Recall': recall_score(test_labels, predicted_labels, average='macro', zero_division=0),
        'F1 Score': f1_score(test_labels, predicted_labels, average='macro', zero_division=0)
    }
    
    # Plotting
    fig, ax = plt.subplots(figsize=(8, 5))  # Adjust size as needed
    sns.barplot(x=list(metrics.keys()), y=list(metrics.values()), ax=ax, palette="viridis")
    
    ax.set_ylim(0, 1)  # Limit the y-axis from 0 to 1 for percentage
    ax.set_ylabel('Score')
    ax.set_title(title)
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.5f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 9), textcoords='offset points')
    
    plt.tight_layout()
    plt.savefig(path + title + '.png')  # Save the plot with the title as the filename
    plt.show()


