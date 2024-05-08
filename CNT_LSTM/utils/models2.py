import torch
import torch.nn as nn
import lightning as L
import numpy as np
from torchmetrics import MeanSquaredError, MeanAbsoluteError, R2Score
from lightning.pytorch.loggers import CSVLogger,TensorBoardLogger

from utils.plots import plot_image
import torchvision
#from torch.optim.lr_scheduler import CosineAnnealingLR

class RECURRENT(L.LightningModule):

    def __init__(self, params):
        super().__init__()

        self.batch_size = params["arch"]["batch_size"]
        self.num_epochs = params["arch"]["num_epochs"]
        self.learning_rate = params["arch"]["learning_rate"]
        self.teacher_forcing = params["arch"]["teacher_forcing"]

        self.model_type = params["model"]["model_type"]
        self.input_size = params["model"]["input_size"]
        self.hidden_size = params["model"]["hidden_size"]
        self.output_size = params["model"]["output_size"]
        self.num_layers = params["model"]["num_layers"] 

        self.input_seq = params["dataset"]["input_seq"]
        self.output_seq = params["dataset"]["output_seq"]

        self.model_type = params["model"]["model_type"]
        if self.model_type == 1:
            self.path_save = params["paths"]["results"]["LSTM"]
        elif self.model_type == 2:
            self.path_save = params["paths"]["results"]["CNN_LSTM"]
        else: 
            self.path_save = params["paths"]["results"]["dense_CNN_LSTM"]



        self.automatic_optimization = False

        # Define Architecture
        if self.model_type == 1:  
            self.lstm = nn.LSTM(input_size=self.input_size, 
                                hidden_size=self.hidden_size, 
                                num_layers=self.num_layers, 
                                batch_first=True)
        elif self.model_type == 2:  
            self.cnn = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2,2)
            )
            self.lstm = nn.LSTM(input_size=4*200*200,  
                                hidden_size=self.hidden_size,
                                num_layers=self.num_layers,
                                batch_first=True)
        elif self.model_type == 3:  
            self.cnn = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2,2),
                nn.Flatten()
            )
            self.lstm = nn.LSTM(input_size=4*200*200, 
                                hidden_size=self.hidden_size,
                                num_layers=self.num_layers,
                                batch_first=True)
            self.dense = nn.Linear(self.hidden_size, self.hidden_size)
        
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
        if self.model_type in [2, 3]:
            inputs = inputs.reshape(1,7,400,400)
            cnn_outputs = []
            for i in range(7):                  
                seq_input = inputs[:, i, :, :].unsqueeze(1)                  
                cnn_out = self.cnn(seq_input)                 
                cnn_out = cnn_out.view(cnn_out.size(0), -1)                 
                cnn_outputs.append(cnn_out)
    
            cnn_out = torch.stack(cnn_outputs, dim=1)
            inputs = cnn_out
        
        lstm_out, hidden = self.lstm(inputs[:, 0:self.input_seq, :], hidden)
        output = self.fc(lstm_out)
        output = torch.unsqueeze(output[:, -1, :], dim=1)
    
        for t in range(self.input_seq, self.input_seq + self.output_seq):
            if self.teacher_forcing == 1 and targets is not None:
                next_input = targets[:, t-1:t, :]
            else:
                next_input = output
            lstm_out, hidden = self.lstm(next_input, hidden)
        
            if self.model_type == 3:                 
                lstm_out = self.dense(lstm_out)

            lstm_out = torch.squeeze(lstm_out, dim=1)
            out = self.fc(lstm_out)
            output = torch.unsqueeze(out, dim=1)

            outputs.append(out)

        outputs = torch.cat(outputs, dim=1)
        outputs = torch.reshape(outputs, (batch_size, self.output_seq, self.output_size))
        return outputs

        
    def emd(self, pred, target):
        pred = pred / pred.sum(dim=-1, keepdim=True)
        target = target / target.sum(dim=-1, keepdim=True)
        cdf_pred = torch.cumsum(pred, dim=-1)
        cdf_target = torch.cumsum(target, dim=-1)
        emd_loss = torch.mean(torch.abs(cdf_pred - cdf_target))
    
        return emd_loss

    def objective(self, preds, labels):

        obj = nn.MSELoss()
        return obj(preds, labels)

    def training_step(self, batch, batch_idx):
         
        x, y = batch
        y_pred = self(x, y)
        losses = [self.objective(y_pred[:, i, :], y[:, i, :]) for i in range(y.shape[1])]
        
        for loss in losses:
            self.manual_backward(loss, retain_graph=True)

        self.optimizers().step()
        self.optimizers().zero_grad()

        total_loss = torch.mean(torch.stack(losses))

        self.log('train_loss', total_loss, batch_size = self.batch_size, on_step=True,
                 on_epoch=True, sync_dist= True)

        log_output = y_pred.view(self.batch_size, self.output_seq, int(np.sqrt(self.output_size)), int(np.sqrt(self.output_size)))
        #tensorboard_logger = next((logger for logger in self.loggers if isinstance(logger, TensorBoardLogger)), None)
        #if tensorboard_logger:
        #    for i in range(self.output_seq):
        #        img = log_output[:, i, :, :].unsqueeze(1)  # Add a channel dimension
        #        img_grid = torchvision.utils.make_grid(img, normalize=True, scale_each=True)
        #        tensorboard_logger.experiment.add_image(f'output_image_{i}', img_grid, self.global_step)

        
        #if self.logger:
        #    for i in range(self.output_seq):
        #        img = log_output[:, i, :, :].unsqueeze(1)  # Add a channel dimension
        #        img_grid = torchvision.utils.make_grid(img, normalize=True, scale_each=True)
        #        self.logger.experiment.add_image(f'output_image_{i}', img_grid, self.global_step)

        #if self.logger:
        #    img_grid = torchvision.utils.make_grid(log_output, normalize=False, scale_each=False)
        #    self.logger.experiment.add_image('output_images', img_grid, self.global_step)


        return loss

    def validation_step(self, batch, batch_idx):
        x,y = batch
        y_pred = self(x,y)
        #plot_image(x, y, y_pred, self.output_seq, (self.output_size, self.output_size), self.input_seq) 

        loss = self.objective(y_pred, y)

        
        
        plot_image(x, y, y_pred, self.output_seq, (self.output_size, self.output_size), self.input_seq, self.path_save) 

        self.log('valid_loss', loss, batch_size = self.batch_size, on_step=True,
                 on_epoch=True, sync_dist= True)

        measures = {"valid_MSE":self.mse, "valid_MAE":self.mae, "valid_emd":self.emd}
        for current_key in measures.keys():
            score = measures[current_key](y_pred, y)
            self.log(current_key, score, batch_size=self.batch_size, on_step=True,
                    on_epoch=True, sync_dist=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

       

