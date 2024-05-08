import sys

from utils.test import run as test
from utils.train import run as train
from utils.general import load_config


def experiment(params):

    if params["experiment"] == 0:

        train(params)

    elif params["experiment"] == 1:

        test(params)

    else:
        raise NotImplementedError


if __name__ == "__main__":

    params = load_config(sys.argv)
    experiment(params)
