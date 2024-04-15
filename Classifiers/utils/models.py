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
        
        from IPython import embed
        embed()
        if name == 'LDA':
            train_samples = train.dataset.samples
            train_samples = train_samples.reshape(train_samples.shape[0],-1)
            model.fit(train_samples, train.dataset.labels)

        # - Calculate training and validation analytics

        #train_preds = model.predict(train.samples)
        #cvalid_preds = model.predict(valid.samples)

        results = {"train": {}, "valid": {}}


