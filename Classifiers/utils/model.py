import pickle
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.multiclass import OneVsRestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from tqdm import tqdm  # For visual progress during training
from utils.measures import compute_metrics
import joblib

def select_models(choices):
    all_models = {}
    
    for index in choices:
        if index == 0:

            # param_grid = {
            #     'estimator__solver': ['svd', 'eigen']
            # }
            # model = GridSearchCV(OneVsRestClassifier(LinearDiscriminantAnalysis()), param_grid, scoring='accuracy')

            model = LinearDiscriminantAnalysis()
            name = "LDA"

        elif index == 1:
 
            param_grid = {
                'estimator__n_neighbors': [5, 7, 31]
            }
            model = GridSearchCV(OneVsRestClassifier(KNeighborsClassifier()), param_grid, scoring='accuracy')
            name = "KNN"
        elif index == 2:
   
            model = GridSearchCV(OneVsRestClassifier(SVC(probability=True)), param_grid, scoring='accuracy')
            name = "SVM"
        else:
            raise NotImplementedError

        all_models[name] = model

    return all_models

def train_sklearn_models(choices, train, valid, measures, path):
    all_models = select_models(choices)
    class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] 
    print("\nSaving Results To: {}\n".format(path))

    for name, model in tqdm(all_models.items(), desc="Training Models"):
        num_samples, height, width = train.samples.shape[0], 28, 28
        X_flat = train.samples.reshape(num_samples, height * width)

        model.fit(X_flat, train.labels)
        
        param_scores = {str(params): mean_score for params, mean_score in zip(model.cv_results_['params'], model.cv_results_['mean_test_score'])}
        train_preds = model.predict(train.samples)
        valid_preds = model.predict(valid.samples)
        train_report = classification_report(train.labels, train_preds, output_dict=True)
        valid_report = classification_report(valid.labels, valid_preds, output_dict=True)

        results = {
            'best_params': model.best_params_,
            'param_scores': param_scores,
            'train_preds': train_preds,
            'valid_preds': valid_preds,
            'train_labels': train.labels,
            'valid_labels': valid.labels,
            'train_report': train_report,
            'valid_report': valid_report
        }

        path_save = f"{path}/{name}2.pkl"
        joblib.dump(results, path_save) 
        
        print(f"Model {name} results saved successfully.")



def train_sklearn_models2(choices, train, valid, measures, path):
    all_models = select_models(choices)
    class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] 
    print("\nSaving Results To: {}\n".format(path))

    for name, model in tqdm(all_models.items(), desc="Training Models"):
        from IPython import embed
        
        num_samples, height, width = train.samples.shape[0], 28, 28
        X_flat = train.samples.reshape(num_samples, height * width)
        from IPython import embed
        embed()
        model.fit(X_flat, train.labels)
        
        param_scores = {str(params): mean_score for params, mean_score in zip(model.cv_results_['params'], model.cv_results_['mean_test_score'])}
        train_preds = model.predict(train.samples)
        valid_preds = model.predict(valid.samples)
        train_report = classification_report(train.labels, train_preds, output_dict=True)
        valid_report = classification_report(valid.labels, valid_preds, output_dict=True)

        results = {
            'best_params': model.best_params_,
            'param_scores': param_scores,
            'train_preds': train_preds,
            'valid_preds': valid_preds,
            'train_labels': train.labels,
            'valid_labels': valid.labels,
            'train_report': train_report,
            'valid_report': valid_report
        }

        path_save = f"{path}/{name}2.pkl"
        with open(path_save, "wb") as writer:
            pickle.dump(results, writer)
        
        print(f"Model {name} results saved successfully.")
        #plot_confusion_matrix(valid.labels, valid_preds, class_names, f"{path}/{name}_confusion_matrix2.png")

def plot_confusion_matrix(labels, preds, class_names, save_path):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(8, 8))
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
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
    plt.savefig(save_path)  
    plt.close(fig) 

#

