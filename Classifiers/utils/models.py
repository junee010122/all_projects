import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from 

def select_models():
    all_models = {
        'LDA': GridSearchCV(OneVsRestClassifier(LinearDiscriminantAnalysis()), 
                            param_grid={'estimator__solver': ['svd', 'lsqr', 'eigen']},
                            cv=5, scoring='accuracy'),
        'KNN': GridSearchCV(OneVsRestClassifier(KNeighborsClassifier()),
                            param_grid={'estimator__n_neighbors': [3, 5, 7, 9, 11]},
                            cv=5, scoring='accuracy'),
        'SVM': GridSearchCV(OneVsRestClassifier(SVC(probability=True)),
                            param_grid={'estimator__C': [0.1, 1, 10, 100], 'estimator__kernel': ['linear', 'poly', 'rbf', 'sigmoid']},
                            cv=5, scoring='accuracy')
    }
    return all_models

def train_sklearn_models(train, valid, class_names, path):
    all_models = select_models()
    
    for name, model in tqdm(all_models.items(), desc="Training Models"):
        model.fit(train.samples, train.labels)
        train_preds = model.predict(train.samples)
        valid_preds = model.predict(valid.samples)

        train_metrics = compute_metrics(train.labels, train_preds)
        valid_metrics = compute_metrics(valid.labels, valid_preds)

        results = {
            'best_params': model.best_params_,
            'param_scores': {str(params): mean_score for params, mean_score in zip(model.cv_results_['params'], model.cv_results_['mean_test_score'])},
            'train_preds': train_preds,
            'valid_preds': valid_preds,
            'train_labels': train.labels,
            'valid_labels': valid.labels,
            'train_metrics': train_metrics,
            'valid_metrics': valid_metrics,
            'train_report': classification_report(train.labels, train_preds, output_dict=True),
            'valid_report': classification_report(valid.labels, valid_preds, output_dict=True)
        }

        path_save = f"{path}/{name}_results.pkl"
        with open(path_save, "wb") as writer:
            pickle.dump(results, writer)

        plot_confusion_matrix(valid.labels, valid_preds, class_names, path, name)

def plot_confusion_matrix(labels, preds, class_names, path, model_name):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(8, 8))
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix for {model_name}')
    fig.colorbar(cax)
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names)

    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        ax.text(j, i, format(cm[i, j], 'd'), ha='center', va='center', 
                color='white' if cm[i, j] > thresh else 'black')

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(f"{path}/{model_name}_confusion_matrix.png")
    plt.close(fig)

# Example usage:
# Assuming 'train' and 'valid' are your datasets and 'path' is your desired save directory
# train_sklearn_models(train, valid, ['Class1', 'Class2', 'Class3', ...], 'model_results')

