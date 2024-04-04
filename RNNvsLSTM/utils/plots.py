import torch
import torch.nn as nn
import lightning as L
from torchmetrics import MeanSquaredError, MeanAbsoluteError, R2Score
from utils.plots import plot_image 


class RecurrentNetwork(L.LightningModule):
    
    """
    Creating a many-one Recurrent Model to solve a regression problem
    """
    def __init__(self, params):

        super().__init__()

        self.model_type = params["model"]["model_type"] 
        
        self.batch_size = params["arch"]["batch_size"]
        self.num_epochs = params["arch"]["num_epochs"]
        self.learning_rate = params["arch"]["learning_rate"]

        self.input_size = params["model"]["input_size"]
        self.output_size = params["model"]["output_size"]
        self.hidden_size = params["model"]["hidden_size"]
        self.num_layers = params["model"]["num_layers"]
        self.hidden = None


        if self.model_type == 0:
            self.recurrent_layer = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        elif model_type == 1:
            self.recurrent_layer = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
       
        self.linear = nn.Linear(self.hidden_size, self.output_size)
    
    def forward(self, x):
        if self.model_type == 'LSTM':
            out, (hn, cn) = self.recurrent_layer(x)
        elif self.model_type == 'RNN':
            out, hn = self.recurrent_layer(x)


        out = self.linear(out[:, -1, :])  # Taking the output of the last sequence step
        return out

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.objective(y_pred, y)  
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)  
        return loss  

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.objective(y_pred, y)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
