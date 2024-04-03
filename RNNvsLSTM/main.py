
import sys
import lightning as L
import torch
import pickle
import os

def run_experiment(params):

    model_type = params["model"]["model_type"]

    train_data, valid_data = load_datasets(params)
    if model_type == 0:
        model = RNN(params)
    else:
        model = LSTM(params)



if __name__ == "__main__":

    params = load_config(sys.argv)

    run_experiment(params)
