import sys

import utils.train as train

from utils.general import load_config


def experiment(params):

    train.run(params)

if __name__ == "__main__":

    params = load_config(sys.argv)

    experiment(params)
