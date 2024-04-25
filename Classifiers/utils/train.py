import lightning as L
import torch
import numpy as np

from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import LearningRateMonitor
from torchvision import transforms
from tqdm import tqdm
import threading
import time

from utils.data import load_data, apply_pca
from utils.models2 import train_sklearn_models
from utils.plots import plot_pca_images


#from utils.models import Network


def run(params):

    path_save = params["paths"]["results"]
    #num_epochs = params["network"]["num_epochs"]
    strategy = params["system"]["gpus"]["strategy"]
    num_devices = params["system"]["gpus"]["num_devices"]
    accelerator = params["system"]["gpus"]["accelerator"]
    choices = params["models"]["choices"]

    # Load: Datasets
    
    train, valid = load_data(params)

    # Diemnsionality reduction : PCA

    train_pca, valid_pca = apply_pca(train, valid)
    
    #plot_pca_images(images_ready, output, model, num_images=5)

    #Create: Model

    measures = params["models"]["measures"]
    choices = params["models"]["choices"]

    
    train_sklearn_models(choices, train_pca, valid_pca, measures, path_save)
    from IPython import embed
    embed()
    # Create: Logger

    exp_logger = CSVLogger(save_dir=path_save)

    # Create: Trainer

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    trainer = L.Trainer(callbacks=[lr_monitor],
                        accelerator=accelerator, strategy=strategy,
                        devices=num_devices, max_epochs=num_epochs,
                        log_every_n_steps=1, logger=exp_logger)

    trainer.fit(model=model, train_dataloaders=train, val_dataloaders=valid)


def spinner(message="Computing"):
    spinner = tqdm(total=None, desc=message, position=0, leave=True)
    while True:
        for cursor in '\\|/-\\|/':
            # Set spinner message
            spinner.set_description_str(f"{message} {cursor}")
            time.sleep(0.1)
            spinner.refresh()  # update the spinner animation
        #if not spinner_flag:
        #    break
    spinner.close()
