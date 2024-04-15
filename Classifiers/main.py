import sys


def experiment(params):

    train.run(params)

if __name__ == "__main__":

    params = load_config(sys.arg)

    experiment(params)
