import lightning as L
import torch

from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import LearningRateMonitor
from torchvision import transforms
from tqdm import tqdm

from utils.data import load_data
from utils.models import train_sklearn_models
from sklearn.decomposition import PCA

#from utils.models import Network

def torch_standardize(x):

    desc = "Data must be numpy formatted: [N, H, W, C], where C = 1 or C = 3"

    assert x.shape[-1] in [1, 3], desc

    num_channels = x.shape[-1]

    t = transforms.Compose([transforms.ToTensor(),
                            transforms.Normalize([0.5] * num_channels,
                                                 [0.5] * num_channels)])

    x = torch.vstack([t(ele).unsqueeze(dim=0)
                     for ele in tqdm(x, desc="Processing")])

    return x.permute(0, 2, 3, 1).numpy()


def run(params):

    #path_save = params["paths"]["results"]
    num_epochs = params["network"]["num_epochs"]
    strategy = params["system"]["gpus"]["strategy"]
    num_devices = params["system"]["gpus"]["num_devices"]
    accelerator = params["system"]["gpus"]["accelerator"]
    choices = params["models"]["choices"]

    # Load: Datasets

    train, valid = load_data(params)
    x = torch_standardize(train.dataset.samples)

    # Diemnsionality reduction : PCA

    #images_ready = x.reshape(x.shape[0], -1)
    #model = PCA(n_components=10)
    #embed()
    #output = model.fit_transform(images_ready)
    
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
