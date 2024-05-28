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
#from torch.optim.lr_scheduler import CosineAnnealingLR

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

        self.input_seq = params["dataset"]["input_seq"]
        self.output_seq = params["dataset"]["output_seq"]

        self.model_type = params["model"]["model_type"]
        if self.model_type == 1:
            self.path_save = params["paths"]["results"]["LSTM"]
        elif self.model_type == 2:
            self.path_save = params["paths"]["results"]["CNN_LSTM"]
        else: 
            self.path_save = params["paths"]["results"]["dense_CNN_LSTM"]


        if self.model_type == 2:  
            self.cnn = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2,2)
            )
            self.lstm = nn.LSTM(input_size=4*155*155,  
                                hidden_size=self.hidden_size,
                                num_layers=self.num_layers,
                                batch_first=True)

        self.fc = nn.Linear(self.hidden_size, self.output_size)

        self.create_validation_measures()

        self.automatic_optimization = False

    def create_validation_measures(self):
        self.mse = MeanSquaredError()
        self.mae = MeanAbsoluteError()
        self.r_squared = R2Score()


    def forward(self, inputs, targets=None):
        batch_size, sequence_length, _ = inputs.size()
        outputs = []
        hidden = None
        inputs = inputs.reshape(1,5,310,310)
        cnn_outputs = []
        inputs[0, 1:self.input_seq+self.output_seq, :, :] = 0
        
        for i in range(5):
            seq_input = inputs[:, i, :, :].unsqueeze(1)
            if self.vertical_slice == 1:
                seq_input = seq_input.transpose(-2,-1)
            cnn_out = self.cnn(seq_input)
            cnn_out = cnn_out.view(cnn_out.size(0), -1)                 
            cnn_outputs.append(cnn_out)
    
        #cnn_out = torch.stack(cnn_outputs, dim=1)
        inputs = cnn_out
        lstm_input = torch.stack(cnn_outputs, dim=1)
        output = None

        for i in range(5):
            if i < self.input_seq:
                input_step = lstm_input[:, i:i+1, :]
            else:
                if output is not None:
                    input_step = output.unsqueeze(1).detach()
                else:
                    continue
        
            lstm_out, hidden = self.lstm(input_step, hidden)
            output = self.fc(lstm_out[:, -1, :])
            outputs.append(output)
        #output = torch.unsqueeze(output[:, -2:, :], dim=1)
        #output = torch.squeeze(output,dim=1)
        outputs = torch.stack(outputs, dim=1)
        #outputs = torch.reshape(output, (batch_size, self.output_seq+self.input_seq, self.output_size))
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
        y_pred = self(x, y)
        
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
        y_pred = self(x,y)
        #plot_image(x, y, y_pred, self.output_seq, (self.output_size, self.output_size), self.input_seq) 

        loss = self.objective(y_pred, y) 
        plot_image(x, y, y_pred, self.output_seq, (self.output_size, self.output_size), self.input_seq, self.path_save) 
        

        buffer = io.BytesIO()
        torch.save(x, "/Users/june/Documents/results/CNN_LSTM/tensors/inputs/inputs.pt")
        torch.save(x,buffer)
        torch.save(y, "/Users/june/Documents/results/CNN_LSTM/tensors/ground_truth/truths.pt")
        torch.save(y,buffer)
        torch.save(y_pred, "/Users/june/Documents/results/CNN_LSTM/tensors/predictions/preds.pt")
        torch.save(y_pred,buffer)


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




if __name__=="__main__":

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














