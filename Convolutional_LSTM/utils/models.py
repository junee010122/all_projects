import torch
import torch.nn as nn
import lightning as L
import numpy as np
from torchmetrics import MeanSquaredError, MeanAbsoluteError, R2Score
#from lightning.pytorchloggers import CSVLogger

from utils.plots import plot_image
import torchvision
from IPython import embed
import io
import os
#from torch.optim.lr_scheduler import CosineAnnealingL
from torchvision.models import resnet50, resnet18, resnet34
from torchvision.models import ResNet50_Weights, ResNet18_Weights, ResNet34_Weights

class RECURRENT(L.LightningModule):

    def __init__(self, params):

        super().__init__()

        self.batch_size = params["arch"]["batch_size"]
        self.num_epochs = params["arch"]["num_epochs"]
        self.learning_rate = params["arch"]["learning_rate"]
        self.teacher_forcing = params["arch"]["teacher_forcing"]

        self.model_type = params["model"]["model_type"]
        self.vertical_slice = params["model"]["slice"]
        self.input_size = params["model"]["input_size"]
        self.hidden_size = params["model"]["hidden_size"]
        self.output_size = params["model"]["output_size"]
        self.num_layers = params["model"]["num_layers"]
        self.lstm_config = params["model"]["lstm_config"]

        self.input_seq = params["dataset"]["input_seq"]
        self.output_seq = params["dataset"]["output_seq"]


        self.model_type = params["model"]["model_type"]
        if self.model_type == 1:
            self.path_save = params["paths"]["results"]["model_1"]
        elif self.model_type == 2:
            self.path_save = params["paths"]["results"]["model_2"]
        elif self.model_type == 3:
            self.path_save = params["paths"]["results"]["model_3"]
        elif self.model_type == 4:
            path_save = params["paths"]["results"]["model_4"]
        elif self.model_type == 5:
            path_save = params["paths"]["results"]["model_5"]
        elif self.model_type == 0:
            self.path_save = params["paths"]["results"]["model_0"]
        
 
        
        self.lstm = nn.LSTM(input_size=self.hidden_size,  
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            batch_first=True)


        self.create_validation_measures()
        self.automatic_optimization = False


        if self.model_type in [0,2,3]:
            if self.model_type == 0:
                backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
            elif self.model_type == 2:
                backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
            else:
                backbone = resnet34(weights=ResNet34_Weights.DEFAULT)

            num_filters = backbone.fc.in_features 
            layers = list(backbone.children())[:-1]
            layers.append(nn.Flatten()) 
            layers.append(nn.Linear(num_filters, self.hidden_size)) 
            self.feature_extractor = nn.Sequential(*layers)


        if self.model_type == 1:
            self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2))
            
        self.fc1 = nn.Linear(self.output_size, self.hidden_size)

        # add 3 more layers for more computation - 32, 1024, power 15, output_size / ReLU
        # self.fc2 = nn.Linear(self.hidden_size, self.output_size)
        
        self.fc2 = nn.Sequential(nn.Linear(self.hidden_size, 2048),
                                 nn.ReLU(),
                                 nn.Linear(2048,self.output_size),
                                 nn.ReLU()
                                 )

    def create_validation_measures(self):
        self.mse = MeanSquaredError()
        self.mae = MeanAbsoluteError()
        self.r_squared = R2Score()


    def forward(self, inputs, targets=None):

        if self.model_type in [0,2,3]:
            inputs=inputs.reshape(1,3,5,310,310)
            batch_size, channel_size, sequence_length, W, H = inputs.size()
            output = []
            for i in range(5):
                seq_input = inputs[:, :, i, :, :]
                feature_output = self.feature_extractor(seq_input) 
                output.append(feature_output)
            combined_output = torch.cat(output, dim=1)
            combined_output = combined_output.resize(1,5,32)


        if self.model_type == 1:
            inputs = inputs.reshape(1,5,310,310)
            batch_size, sequence_length, W, H = inputs.size()
            cnn_outputs = []

            for i in range(5):
                seq_input = inputs[:,i,:,:].unsqueeze(1)
                if self.vertical_slice == 1:
                    seq_input = seq_input.transpose(-2,-1)
                cnn_out = self.cnn(seq_input)
                cnn_out = cnn_out.view(cnn_out.size(0),-1)
                cnn_outputs.append(cnn_out)
            cnn_out = torch.stack(cnn_outputs, dim=1)
            combined_output = cnn_out
            combined_output = self.fc1(combined_output)    
            
        outputs = []
        hidden = None
        
        lstm_input=combined_output
        
        for i in range(5):
            if i<self.input_seq+self.output_seq:
                input_step = lstm_input[:, i:i+1, :]
            else:
                if output is not None:
                    input_step = output.unsqueeze(1).detach()

                else:
                    continue    
            if self.lstm_config == 0:
                lstm_out, hidden = self.lstm(input_step, hidden)
                output = lstm_out
            elif self.lstm_config == 1:
                current_input = input_step
                for _ in range(5):  
                    lstm_out, hidden = self.lstm(current_input, hidden)
                    current_input = lstm_out
                output = lstm_out
            elif self.lstm_config == 2:
                current_input = input_step
                lstm_out, hidden = self.lstm(current_input, hidden)
                zero_input = torch.zeros_like(current_input)
                for _ in range(4):  # Continue for the remaining steps, total steps 5
                    lstm_out, hidden = self.lstm(zero_input, hidden)
                output = lstm_out
            output = self.fc2(lstm_out)
            outputs.append(output)
        outputs = torch.stack(outputs, dim=1)
        outputs = torch.sigmoid(outputs) 
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
        
        if self.model_type in [0,2,3]:
            x = x.repeat(3,1,1)
            x = x.unsqueeze(dim=0)
        y_pred = self(x, y)
        y_pred = y_pred.squeeze(dim=-2)

        
        losses = self.objective(y_pred, y)
        
        losses = [self.objective(y_pred, y) for i in range(y.shape[1])]
        
        for loss in losses: # change this part so that backprop makes use of all outputs
            self.manual_backward(loss, retain_graph=True)

        self.optimizers().step()
        self.optimizers().zero_grad()
        total_loss = torch.mean(torch.stack(losses))

        self.log('train_loss', total_loss, batch_size = self.batch_size, on_step=True,
                 on_epoch=True, sync_dist= True)

        #log_output = y_pred.view(self.batch_size, self.output_seq, int(np.sqrt(self.output_size)), int(np.sqrt(self.output_size)))

        return total_loss

    def validation_step(self, batch, batch_idx):
        x,y = batch
        
        if self.model_type in [0,2,3]:
            x = x.repeat(3,1,1)
            x = x.unsqueeze(dim=0)
        y_pred = self(x,y)
        #plot_image(x, y, y_pred, self.output_seq, (self.output_size, self.output_size), self.input_seq) 

        y_pred = y_pred.squeeze(dim=-2)
        loss = self.objective(y_pred, y) 
        plot_image(x, y, y_pred, self.output_seq, (self.output_size, self.output_size), self.input_seq, self.path_save) 


        buffer = io.BytesIO()

        tensor_dict = {
            'inputs.pt': x,
            'truths.pt': y,
            'preds.pt': y_pred
            }

        for name, tensor in tensor_dict.items():
            tensor_path = os.path.join(self.path_save, name)  # Construct the full path for the tensor
            torch.save(tensor, tensor_path)
            torch.save(tensor,buffer)


        self.log('valid_loss', loss, batch_size = self.batch_size, on_step=True,
                 on_epoch=True, sync_dist= True)

        measures = {"valid_mse":self.mse, "valid_mae":self.mae, "valid_emd":self.emd}
        for current_key in measures.keys():
            score = measures[current_key](y_pred, y)
            self.log(current_key, score, batch_size=self.batch_size, on_step=True,
                    on_epoch=True, sync_dist=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer













