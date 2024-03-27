import numpy as np
import torch
from torch.utils.data import DataLoader

from PIL import Image

class Dataset:

    def __init__(self, samples, labels, params):
        self.samples = samples
        self.labels = labels
        
        self.input_seq = params["dataset"]["input_seq"]
        self.output_seq = params["dataset"]["output_seq"]
        self.resize = params["dataset"]["resize"]


    def format(self, all_files):
    
        sequence=[]
        for current_file in all_files:
            image = Image.open(current_file).convert("L")
            size = int(image.size[0]* self.resize)
            image = np.asarray(image.resize((size,size))).reshape(-1)
            image = image / 255
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

    all_samples, all_labels = [],[]

    for i in range(len(data)):

        if i>len(data) - out_size - in_size:
            continue

        all_samples.append(data[i:i+in_size])
        all_labels.append(data[i+in_size:i + in_size + out_size])

    data = Dataset(all_samples, all_labels, params)
    data = torch.utils.data.DataLoader(data, shuffle=True,
                                       batch_size=batch_size,
                                       num_workers=num_workers)
   
    return data



    


        
