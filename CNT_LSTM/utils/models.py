import torch
import torch.nn as nn
import lightning as L
from torchmetrics import MeanSquaredError, MeanAbsoluteError, R2Score
from utils.plots import plot_image 
#from torch.optim.lr_scheduler import CosineAnnealingLR

class LSTM(L.LightningModule):

    def __init__(self, params):
        super().__init__()
        
        self.batch_size = params["arch"]["batch_size"]
        self.num_epochs = params["arch"]["num_epochs"]
        self.learning_rate = params["arch"]["learning_rate"]
        self.teacher_forcing = params["arch"]["teacher_forcing"]

        self.input_size = params["model"]["input_size"]
        self.hidden_size = params["model"]["hidden_size"]
        self.output_size = params["model"]["output_size"]
        self.num_layers = params["model"]["num_layers"]

        self.input_seq = params["dataset"]["input_seq"]
        self.output_seq = params["dataset"]["output_seq"]


        # Define Architecture

        self.lstm = nn.LSTM(input_size=self.input_size, 
                            hidden_size=self.hidden_size, 
                            num_layers=self.num_layers, 
                            batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.output_size)

        
        self.create_validation_measures()

    def create_validation_measures(self):

        self.mse = MeanSquaredError()
        self.mae = MeanAbsoluteError()
        self.r_squared = R2Score()


    def forward(self, inputs, targets=None):
        batch_size, sequence_length, _ = inputs.size()
        outputs = []

        hidden = None
        from IPython import embed
        

        lstm_out, hidden = self.lstm(inputs, hidden)
        out = self.fc(lstm_out[:, -1, :])
        
        #for t in range(self.input_seq):
        #    lstm_out, hidden = self.lstm(inputs[:, t:t+1, :], hidden)
        #    if t == self.input_seq-1: 
        #        out = self.fc(lstm_out)
        #        outputs.append(out)



        for t in range(1, targets.size(1)):  
            if self.teacher_forcing == 1 and self.training and targets is not None:
                next_input = outputs[-1]

            else:
                next_input = inputs[:, t-1:t, :]
  
            lstm_out, hidden = self.lstm(next_input, hidden)
            out = self.fc(lstm_out)
            outputs.append(out)
    

        outputs = torch.cat(outputs, dim=1)
        
        return outputs
    
    def Emd(self, pred, target):
        pred = pred / pred.sum(axis=1, keepdims=True)
        target = target / target.sum(axis=1, keepdims=True)
    
        cdf_pred = torch.cumsum(pred, dim=1)
        cdf_target = torch.cumsum(target, dim=1)
        emd_loss = torch.sum(torch.abs(cdf_pred - cdf_target), axis=1)
        emd_loss = emd_loss.mean()
        return emd_loss

    def objective(self, preds, labels):

        obj = nn.MSELoss()

        return obj(preds, labels)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x, y)
        loss = self.objective(y_pred, y)

        self.log('train_loss', loss, batch_size = self.batch_size, on_step=True,
                 on_epoch=True, sync_dist= True)

        return loss

    def validation_step(self, batch, batch_idx):
        x,y = batch
        y_pred = self(x,y)
        loss = self.objective(y_pred, y)

        plot_image(y, y_pred, self.output_seq, (self.output_size, self.output_size)) 

        self.log('valid_loss', loss, batch_size = self.batch_size, on_step=True,
                 on_epoch=True, sync_dist= True)

        measures = {"valid_MSE":self.mse, "valid_MAE":self.mae, "valid_EMD": self.Emd}
        for current_key in measures.keys():
            score = measures[current_key](y_pred, y)
            self.log(current_key, score, batch_size=self.batch_size, on_step=True,
                    on_epoch=True, sync_dist=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

       

