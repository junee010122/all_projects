import pickle

from tqdm import tqdm

from sklearn.multiclass import OneVsRestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

#from utils.measures import comparison
from sklearn.ensemble import VotingClassifier

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


def train_sklearn_models(choices, train, valid, measures, path):
    """
    Purpose:
    - Train machine learning models and save their results

    Arguments:
    - path (str): path to save model results
    - choices (list[int]): model indices
    - measures (list[int]): measure indices
    - data (Dataset): machine learning dataset
    """

    # Select: Algorithms

    all_models = select_models(choices)

    # Train: Algorithms

    print("\nSaving Results To: %s\n" % path)

    for name in tqdm(all_models.keys(), "Training Models"):

        # - Update save path

        path_save = path + "/%s.pkl" % name

        # - Train curernt model on training dataset
        model = all_models[name]
        
        estimators = []
        for name, model in tqdm(all_models.items(), desc="Training Models"):
            model.fit(train.samples, train.labels)
            train_preds = model.predict(train.samples)
            valid_preds = model.predict(valid.samples)

            train_metrics = compute_metrics(train.labels, train_preds)
            valid_metrics = compute_metrics(valid.labels, valid_preds)

            # Append model to estimators for voting
            estimators.append((name, model))

            results = {
                'train_metrics': train_metrics,
                'valid_metrics': valid_metrics,
                'train_preds': train_preds,
                'valid_preds': valid_preds
            }
            path_save = f"{path}/{name}.pkl"
            with open(path_save, "wb") as writer:
                pickle.dump(results, writer)

            plot_confusion_matrix(valid.labels, valid_preds, class_names)
            # Save individual model results
            estimators.append((name, model))

        ensemble = VotingClassifier(estimators=estimators, voting='hard')
        ensemble.fit(train.samples, train.labels)
        ensemble_train_preds = ensemble.predict(train.samples)
        ensemble_valid_preds = ensemble.predict(valid.samples)

        # Evaluate ensemble
        ensemble_train_metrics = compute_metrics(train.labels, ensemble_train_preds)
        ensemble_valid_metrics = compute_metrics(valid.labels, ensemble_valid_preds)
        ensemble_results = {
            'train_metrics': ensemble_train_metrics,
            'valid_metrics': ensemble_valid_metrics
        }


        ensemble_path = f"{path}/ensemble.pkl"
        with open(ensemble_path, "wb") as writer:
            pickle.dump(ensemble_results, writer)

        print("Models and ensemble trained and evaluated successfully.")




