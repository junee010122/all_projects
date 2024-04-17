import pickle

from tqdm import tqdm

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


def select_models(choices):
    """
    Purpose:
    - Select a set of models for analysis

    Arguments:
    - choices (list[int]): model indices

    Returns:
    - (dict[str, any]): selected models
    """

    all_models = {}

    for index in choices:

        if index == 0:
            model = LinearDiscriminantAnalysis()
            name = "LDA"

        elif index == 1:
            model = KNeighborsClassifier()
            name = "KNN"

        elif index == 2:
            model = SVC()
            name = "SVC"

        else:
            raise NotImplementedError

        all_models[name] = model

    return all_models


def train_sklearn_models(choices, train, valid):
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

    #print("\nSaving Results To: %s\n" % path)

    for name in tqdm(all_models.keys(), "Training Models"):

        # - Update save path

        #path_save = path + "/%s.pkl" % name

        # - Train curernt model on training dataset
        model = all_models[name]

        train_samples = train.dataset.samples
        train_samples = train_samples.reshape(train_samples.shape[0],-1)
        model.fit(train_samples, train.dataset.labels)

        # - Calculate training and validation analytics
        train_preds = model.predict(train_samples)
        valid_samples = validation.dataset.samples
        valid_samples = valid_samples.reshape(valid_samples.shape[0], -1)
        valid_preds = model.predict(valid_samples)

        results = {"train": {}, "valid": {}}

        for m in measures:

            tag, t_measures = comparison(train.labels, train_preds, choice=m)
            tag, v_measures = comparison(valid.labels, valid_preds, choice=m)

            # - Organize analytics

            results["train"][tag] = t_measures
            results["valid"][tag] = v_measures

        results = {"name": name, "model": model, "results": results}

        # - Save analytics

        with open(path_save, "wb") as writer:
            pickle.dump(results, writer)



