
import sys
import lightning as L
import torch
import pickle
import os

from lightning.pytorch.loggers import CSVLogger,TensorBoardLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

from utils.general import load_config, load_datasets
from utils.models import RECURRENT
from IPython import embed
#from utils.plots import make_video


def run_experiment(params):


    model_type = params["model"]["model_type"]
    path_save = params["paths"]["results"]
    num_epochs = params["arch"]["num_epochs"]
    strategy = params["system"]["gpus"]["strategy"]
    num_devices = params["system"]["gpus"]["num_devices"]
    accelerator = params["system"]["gpus"]["accelerator"]

    if model_type == 1:
        path_save = params["paths"]["results"]["model_1"]
    elif model_type == 2:
        path_save = params["paths"]["results"]["model_2"]
    elif model_type == 3:
        path_save = params["paths"]["results"]["model_3"]
    elif model_type == 4:
        path_save = params["paths"]["results"]["model_4"]
    elif model_type == 5:
        path_save = params["paths"]["results"]["model_5"]
    elif model_type == 0:
        path_save = params["paths"]["results"]["model_0"]



        
    
    # Load Data
    train_data, valid_data = load_datasets(params)   

    # Create RECURRENT Model
    checkpoint_path = os.path.join(path_save, "checkpoint")
    #checkpoint_callback  = ModelCheckpoint(save_top_k=-1,
    #                                       every_n_epochs=1) 
    model = RECURRENT(params)

    exp_logger = CSVLogger(save_dir=path_save)
    #pred_logger = TensorBoardLogger(save_dir=path_save)
    lr_monitor = LearningRateMonitor(logging_interval="epoch")    

    # Create Trainer
    #trainer = L.Trainer(callbacks=[lr_monitor,checkpoint_callback],
    trainer = L.Trainer(callbacks=lr_monitor,
                        accelerator=accelerator, strategy=strategy,
                        devices=num_devices, max_epochs=num_epochs,
                        log_every_n_steps=1, logger=exp_logger, default_root_dir=checkpoint_path)
                        #log_every_n_steps=1, logger=[exp_logger, pred_logger], default_root_dir=checkpoint_path,)
    
    # Train & Evaluate Model
    model.train()
    embed()
    trainer.fit(model=model, train_dataloaders=train_data, val_dataloaders=valid_data)
    

    # Plot results - compare predicted vs truth

    

if __name__ == "__main__":

    # Load Config File
    params = load_config(sys.argv)
    
    # Launch Experiment
    run_experiment(params)

    
