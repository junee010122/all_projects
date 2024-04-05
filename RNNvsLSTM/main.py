
import sys
import lightning as L
import torch
import pickle
import os

from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import LearningRateMonitor
from sklearn.model_selection import train_test_split

from utils.data import preprocess_secom_data, load_datasets
from utils.general import load_config
from utils.models import RecurrentNetwork


def run_experiment(params):

    model_type = params["model"]["model_type"]
    data_path = params["paths"]["data_path"]
    save_path = params["paths"]["results"]
    labels_path = params["paths"]["labels_path"]
    num_epochs = params["arch"]["num_epochs"]
    strategy = params["system"]["gpus"]["strategy"]
    num_devices = params["system"]["gpus"]["num_devices"]
    accelerator = params["system"]["gpus"]["accelerator"]

    data, labels = preprocess_secom_data(data_path, labels_path)
    
    train_samples, test_samples, train_labels, test_labels = train_test_split(
        data, labels, test_size=0.2, random_state=42)

    datasets = load_datasets(train_samples, train_labels, test_samples, test_labels)
    from IPython import embed
    model = RecurrentNetwork(params)
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    exp_logger = CSVLogger(save_dir = save_path)
    trainer = L.Trainer(callbacks=[lr_monitor],
                        accelerator=accelerator, strategy=strategy,
                        devices=num_devices, max_epochs=num_epochs,
                        log_every_n_steps=1, logger=exp_logger)

    model.train()
    trainer.fit(model=model, train_dataloaders=datasets['train'], val_dataloaders=datasets['test'])


if __name__ == "__main__":

    params = load_config(sys.argv)

    run_experiment(params)
