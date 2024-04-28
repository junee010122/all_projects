import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt
import numpy as np

def select_models(choices):
    all_models = {}

    for index in choices:
        if index == 0:
            # LDA
            param_grid = {'solver': ['svd', 'lsqr', 'eigen'],
                          'shrinkage': ['auto', None, 0.1, 0.5, 0.9],
                          'n_components': [None, 1, 2, 3, 4, 5, 10, 20]}
            model = GridSearchCV(OneVsRestClassifier(LinearDiscriminantAnalysis()), param_grid, cv=5, scoring='accuracy')
            name = "LDA"
        elif index == 1:
            # KNN
            for n_neighbors in [3, 5, 7, 9, 11]:
                param_grid = {'estimator__n_neighbors': [n_neighbors]}
                knn_model = GridSearchCV(OneVsRestClassifier(KNeighborsClassifier()), param_grid, cv=5, scoring='accuracy')
                all_models[f"KNN_{n_neighbors}"] = knn_model
            continue
        elif index == 2:
            # SVM
            param_grid = {'estimator__C': [0.1, 1, 10, 100],
                          'estimator__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                          'estimator__gamma': ['scale', 'auto']}
            model = GridSearchCV(OneVsRestClassifier(SVC(probability=True)), param_grid, cv=5, scoring='accuracy')
            name = "SVM"
        else:
            raise NotImplementedError
        
        all_models[name] = model

    return all_models

def plot_confusion_matrix(labels, preds, class_names):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(8, 8))
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    fig.colorbar(cax)
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45)
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
    plt.show()

def train_sklearn_models(choices, train, valid, measures, path):
    all_models = select_models(choices)
    class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    print("\nSaving Results To: %s\n" % path)
    for name, model in all_models.items():
        # Train the model on the training dataset
        model.fit(train.samples, train.labels)

        # Extract hyperparameters and corresponding scores
        param_scores = {}
        for params, score in zip(model.cv_results_['params'], model.cv_results_['mean_test_score']):
            param_scores[str(params)] = score

        # Organize results including predictions, ground truth, and reports
        results = {
            'best_params': model.best_params_,
            'param_scores': param_scores,
            'train_report': classification_report(train.labels, model.predict(train.samples), output_dict=True),
            'valid_report': classification_report(valid.labels, model.predict(valid.samples), output_dict=True)
        }
        
        # Save the results to a pickle file
        path_save = f"{path}/{name}.pkl"
        with open(path_save, "wb") as writer:
            pickle.dump(results, writer)
        
        print(f"Model {name} results saved successfully.")
        plot_confusion_matrix(valid.labels, model.predict(valid.samples), class_names)

    print("Models trained and evaluated successfully.")

