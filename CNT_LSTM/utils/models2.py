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

        self.automatic_optimization = False

        # Define Architecture
        if self.model_type == 1:  # Basic LSTM
            self.lstm = nn.LSTM(input_size=self.input_size, 
                                hidden_size=self.hidden_size, 
                                num_layers=self.num_layers, 
                                batch_first=True)
        elif self.model_type == 2:  # CNN-LSTM
            self.cnn = nn.Sequential(
                nn.Conv1d(in_channels=self.input_size, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2),
                nn.Flatten()
            )
            self.lstm = nn.LSTM(input_size=64,  # Adjusted based on CNN output
                                hidden_size=self.hidden_size,
                                num_layers=self.num_layers,
                                batch_first=True)
        elif self.model_type == 3:  # CNN-LSTM-Dense
            self.cnn = nn.Sequential(
                nn.Conv1d(in_channels=self.input_size, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2),
                nn.Flatten()
            )
            self.lstm = nn.LSTM(input_size=64,  # Adjusted based on CNN output
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

        

    #def on_epoch_start(self):
    #    self.hidden = self.init_hidden(self.batch_size)
    
    #def init_hidden(self, batch_size):
    
    #    return (torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device),
    #            torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device))
    def forward(self, inputs, targets=None):
        batch_size, sequence_length, _ = inputs.size()
        outputs = []
        hidden = None

        if self.model_type in [2, 3]:  # CNN-LSTM or CNN-LSTM-Dense
            inputs = inputs.permute(0, 2, 1)  # Adjust shape for Conv1D
            inputs = self.cnn(inputs)
            inputs = inputs.view(batch_size, -1, 64)  # Adjust shape for LSTM

        if self.model_type == 3:  # CNN-LSTM-Dense
            lstm_out, hidden = self.lstm(inputs, hidden)
            lstm_out = self.dense(lstm_out)
        else:
            lstm_out, hidden = self.lstm(inputs, hidden)

        output = self.fc(lstm_out[:, -1, :])
        outputs.append(output.unsqueeze(1))
        outputs = torch.cat(outputs, dim=1)
        return outputs

    def forward(self, inputs, targets=None):
        batch_size, sequence_length, _ = inputs.size()
        outputs = []
        hidden = None

        if self.model_type in [2, 3]:  # CNN-LSTM or CNN-LSTM-Dense
            inputs = inputs.permute(0, 2, 1)  # Adjust shape for Conv1D
            inputs = self.cnn(inputs)
            inputs = inputs.view(batch_size, -1, 64)  # Adjust shape for LSTM input

        lstm_out, hidden = self.lstm(inputs, hidden)
        
        if self.model_type == 3:  # CNN-LSTM-Dense
            lstm_out = self.dense(lstm_out)

        output = self.fc(lstm_out[:, -1, :])
        outputs.append(output.unsqueeze(1))

        # Teacher forcing logic
        for t in range(1, self.output_seq):
            if self.teacher_forcing and targets is not None:
                next_input = targets[:, t-1:t, :]
            else:
                next_input = output

            if self.model_type in [2, 3]:  # Adjust input shape for CNN
                next_input = next_input.permute(0, 2, 1)
                next_input = self.cnn(next_input)
                next_input = next_input.view(batch_size, -1, 64)

            lstm_out, hidden = self.lstm(next_input, hidden)
            
            if self.model_type == 3:  # CNN-LSTM-Dense
                lstm_out = self.dense(lstm_out)

            output = self.fc(lstm_out[:, -1, :])
            outputs.append(output.unsqueeze(1))

        outputs = torch.cat(outputs, dim=1)
        return outputs

        
    def emd(self, pred, target):
        # Normalizing the predictions and target to make them proper probability distributions
        pred = pred / pred.sum(dim=-1, keepdim=True)
        target = target / target.sum(dim=-1, keepdim=True)
    
        # Calculating the cumulative distribution functions (CDFs) for both predictions and target
        cdf_pred = torch.cumsum(pred, dim=-1)
        cdf_target = torch.cumsum(target, dim=-1)
    
        # Calculating the EMD loss as the mean of the absolute differences between the CDFs
        emd_loss = torch.mean(torch.abs(cdf_pred - cdf_target))
    
        return emd_loss

    def objective(self, preds, labels):

        obj = nn.MSELoss()
        return obj(preds, labels)

    def training_step(self, batch, batch_idx):
         
        x, y = batch
        y_pred = self(x, y)

        #plot_image(x, y, y_pred, self.output_seq, (self.output_size, self.output_size), self.input_seq) 
        #from IPython import embed
        #embed()
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
        
        plot_image(x, y, y_pred, self.output_seq, (self.output_size, self.output_size), self.input_seq) 

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

       

