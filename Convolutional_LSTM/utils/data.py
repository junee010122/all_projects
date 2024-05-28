import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from PIL import Image
from IPython import embed

class Dataset:

    def __init__(self, samples, labels, params):
        self.samples = samples
        self.labels = labels
        
        self.input_seq = params["dataset"]["input_seq"]
        self.output_seq = params["dataset"]["output_seq"]
        self.resize = params["dataset"]["resize"]
        self.model_type = params["model"]["model_type"]


    def format(self, all_files):
    
        sequence=[]
        for current_file in all_files:
            image = Image.open(current_file).convert("L")
            size = int(image.size[0]* self.resize)
            image = np.asarray(image.resize((size,size))).reshape(-1)
            image = image / 255
            image = np.abs(image - 1)
            sequence.append(image)

        sequence = np.asarray(sequence)
        
        return sequence.astype(np.float32)

    def __getitem__(self, index):

        sample_seq_files, label_seq_files = self.samples[index], self.labels[index]
        sample = self.format(sample_seq_files)
        label = self.format(label_seq_files)
        
        return sample, label

    def __len__(self):

        return len(self.samples)

def convert_data(data, params):
    
    in_size = params["dataset"]["input_seq"]
    out_size = params["dataset"]["output_seq"]
    batch_size = params["arch"]["batch_size"]
    num_workers = params["system"]["num_workers"]
    model_type = params["model"]["model_type"]

    all_samples, all_labels = [],[]

    for i in range(len(data)-5+1):

        all_labels.append(data[i+1:i + in_size + out_size+1])
        all_samples.append(data[i:i+in_size+out_size])
    del(all_labels[-1])
    del(all_samples[-1])
    dataset = Dataset(all_samples, all_labels, params)

    total_size = len(dataset)
    train_size = int(total_size * 0.8)
    valid_size = total_size - train_size 

    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

   
    return train_loader, valid_loader


if __name__ == "__main__":
    import yaml
    from general import load_config, load_datasets
    import sys
    from models import RECURRENT
    from lightning.pytorch.callbacks import LearningRateMonitor
    import lightning as L
    from lightning.pytorch.loggers import CSVLogger


    params = load_config(sys.argv)
    train_data, valid_data = load_datasets(params)

    model = RECURRENT(params)
    num_epochs = 10
    lr_monitor = LearningRateMonitor(logging_interval="epoch")    
    exp_logger = CSVLogger(save_dir='/Users/june/Desktop/playground/utils')

    trainer = L.Trainer(callbacks=lr_monitor,
                        accelerator='cpu', strategy='auto',
                        devices=1, max_epochs=num_epochs,
                        log_every_n_steps=1, logger=exp_logger)

    #lr_monitor = LearningRateMonitor(logging_interval="epoch")    
    trainer = L.Trainer(max_epochs=num_epochs)

    




    


        
