import lightning as L
import torch

from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import LearningRateMonitor
from torchvision import transforms
from tqdm import tqdm

from utils.data import load_data
from utils.models import train_sklearn_models
from utils.plots import plot_pca_images 
from sklearn.decomposition import PCA

#from utils.models import Network

def run(params):

    #path_save = params["paths"]["results"]
    num_epochs = params["network"]["num_epochs"]
    strategy = params["system"]["gpus"]["strategy"]
    num_devices = params["system"]["gpus"]["num_devices"]
    accelerator = params["system"]["gpus"]["accelerator"]
    choices = params["models"]["choices"]

    # Load: Datasets
    
    train, valid = load_data(params)

    # Diemnsionality reduction : PCA

    images_ready = train.dataset.samples.reshape(train.dataset.samples.shape[0], -1)

    model = PCA(n_components=3)
    output = model.fit_transform(images_ready)
    plot_pca_images(images_ready, output, model, num_images=5)

    # Create: Model
    train_sklearn_models(choices, train, valid)
    model = Network(params)

    # Create: Logger

    exp_logger = CSVLogger(save_dir=path_save)

    # Create: Trainer

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    trainer = L.Trainer(callbacks=[lr_monitor],
                        accelerator=accelerator, strategy=strategy,
                        devices=num_devices, max_epochs=num_epochs,
                        log_every_n_steps=1, logger=exp_logger)

    trainer.fit(model=model, train_dataloaders=train, val_dataloaders=valid)
