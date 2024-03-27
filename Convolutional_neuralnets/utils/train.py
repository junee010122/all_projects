import lightning as L

from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import LearningRateMonitor

from utils.data import load_data
from utils.models import Network


def run(params):

    path_save = params["paths"]["results"]
    num_epochs = params["network"]["num_epochs"]
    strategy = params["system"]["gpus"]["strategy"]
    num_devices = params["system"]["gpus"]["num_devices"]
    accelerator = params["system"]["gpus"]["accelerator"]

    # Load: Datasets

    train, valid = load_data(params)

    # Create: Model

    model = Network(params)

    # Create: Logger

    exp_logger = CSVLogger(save_dir=path_save)

    # Create: Trainer

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    trainer = L.Trainer(callbacks=[lr_monitor],
                        accelerator=accelerator, strategy=strategy,
                        devices=num_devices, max_epochs=num_epochs,
                        log_every_n_steps=1, logger=exp_logger)

    # Train: Model
    from IPython import embed
    embed()
    trainer.fit(model=model, train_dataloaders=train, val_dataloaders=valid)
