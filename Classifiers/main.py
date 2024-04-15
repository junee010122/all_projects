import sys

import utils.train as train
#import utils.evaluate as evalutate

from utils.general import load_config


def experiment(params):


    train_model = params["models"]["train"]

    if train_model:
        train.run(params)
    #else:
    #    evaluate.run(params)

if __name__ == "__main__":

    params = load_config(sys.argv)

    experiment(params)
