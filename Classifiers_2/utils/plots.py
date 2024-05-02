import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


def plot_decision_boundary(X, y, classifier, title):
    h = .02  # Step size in the mesh
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])  # Light colors for decision regions
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])  # Bold colors for data points

    # Plot the decision boundary. For that, we will assign a color to each point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    from IPython import embed
    embed()
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap=cmap_light)

    # Plot the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

def plot_projected(data, labels, class_colors=None):
    x_values = data[:, 0]
    y_values = data[:, 1]

    # Create scatter plot
    plt.figure(figsize=(8, 6))

    if class_colors is None:
        class_colors = {
            0: 'red',
            1: 'blue',
            2: 'green'
        }

    # Scatter plot for each class
    for class_label, color in class_colors.items():
        class_indices = labels == class_label
        plt.scatter(x_values[class_indices], y_values[class_indices], c=color, label=f'Class {class_label}')

    plt.title('Projected_data')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True)
    plt.show()
    
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
    #ax.set_title(title)
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.5f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 9), textcoords='offset points')
    
    plt.tight_layout()
    ax.title.set_position([.5, 1.05])
    plt.savefig(path + title + '.png')  # Save the plot with the title as the filename
    #plt.show()


