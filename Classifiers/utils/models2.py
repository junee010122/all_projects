
import pickle

from tqdm import tqdm

from sklearn.multiclass import OneVsRestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np

#from utils.measures import comparison
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix


def select_models(choices):


    all_models = {}

    for index in choices:

        if index == 0:
            model = OneVsRestClassifier(LinearDiscriminantAnalysis())
            name = "LDA"

        elif index == 1:
            model = OneVsRestClassifier(KNeighborsClassifier())
            name = "KNN"

        elif index == 2:
            model = OneVsRestClassifier(SVC(probability=True))
            name = "SVC"

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
    estimators = []
    for name, model in tqdm(all_models.items(), desc="Training Models"):
        # Train the model on the training dataset
        model.fit(train.samples, train.labels)
        train_preds = model.predict(train.samples)
        valid_preds = model.predict(valid.samples)

        # Calculate the classification report for both training and validation sets
        train_report = classification_report(train.labels, train_preds, output_dict=True)
        valid_report = classification_report(valid.labels, valid_preds, output_dict=True)

        # Organize results including predictions, ground truth, and reports
        results = {
            'train_preds': train_preds,
            'valid_preds': valid_preds,
            'train_labels': train.labels,
            'valid_labels': valid.labels,
            'train_report': train_report,
            'valid_report': valid_report
        }
        
        # Save the results to a pickle file
        path_save = f"{path}/{name}.pkl"
        with open(path_save, "wb") as writer:
            pickle.dump(results, writer)
        
        print(f"Model {name} results saved successfully.")
        plot_confusion_matrix(valid.labels, valid_preds, class_names)

    
    # After training individual models
    ensemble = VotingClassifier(estimators=estimators, voting='hard')
    ensemble.fit(train.samples, train.labels)
    ensemble_train_preds = ensemble.predict(train.samples)
    ensemble_valid_preds = ensemble.predict(valid.samples)

    ensemble_train_report = classification_report(train.labels, ensemble_train_preds, output_dict=True)
    ensemble_valid_report = classification_report(valid.labels, ensemble_valid_preds, output_dict=True)

    # Evaluate ensemble
    ensemble_results = {
        'train_preds': ensemble_train_preds,
        'valid_preds': ensemble_valid_preds,
        'train_labels': train.labels,
        'valid_labels': valid.labels,
        'train_report': ensemble_train_report,
        'valid_report': ensemble_valid_report
    }

    ensemble_path = f"{path}/ensemble.pkl"
    with open(ensemble_path, "wb") as writer:
        pickle.dump(ensemble_results, writer)

    print("Ensemble model trained and evaluated successfully.")

    # Optionally plot confusion matrices for ensemble predictions
    plot_confusion_matrix(train.labels, ensemble_train_preds, class_names)
    plot_confusion_matrix(valid.labels, ensemble_valid_preds, class_names)

    print("Models and ensemble trained and evaluated successfully.")

