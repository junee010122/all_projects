
import sys
import lightning as L
import torch
import pickle
import os

from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import LearningRateMonitor

from utils.general import load_config, load_datasets
from utils.models import LSTM
#from utils.plots import make_video


def run_experiment(params):

    path_save = params["paths"]["results"]
    num_epochs = params["arch"]["num_epochs"]
    strategy = params["system"]["gpus"]["strategy"]
    num_devices = params["system"]["gpus"]["num_devices"]
    accelerator = params["system"]["gpus"]["accelerator"]
    
    # Load Data
    train_data, valid_data = load_datasets(params)   
    from IPython import embed

    # Create LSTM Model
    model = LSTM(params)

    exp_logger = CSVLogger(save_dir=path_save)
    lr_monitor = LearningRateMonitor(logging_interval="epoch")    

    # Create Trainer
    trainer = L.Trainer(callbacks=[lr_monitor],
                        accelerator=accelerator, strategy=strategy,
                        devices=num_devices, max_epochs=num_epochs,
                        log_every_n_steps=1, logger=exp_logger)
    
    # Train & Evaluate Model
    model.train()
    trainer.fit(model=model, train_dataloaders=train_data, val_dataloaders=valid_data)
    

    # Plot results

    

if __name__ == "__main__":

    # Load Config File
    params = load_config(sys.argv)
    
    # Launch Experiment
    run_experiment(params)

    
