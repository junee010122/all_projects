import torch
import torch.nn as nn
import lightning as L
from torchmetrics import Accuracy, Precision, Recall, F1Score

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
        self.num_classes = params["model"]["num_class"]
        self.output_size = params["model"]["output_size"]
        self.hidden_size = params["model"]["hidden_size"]
        self.num_layers = params["model"]["num_layers"]


        if self.model_type == 0:
            # the input size as to be the same as the feature size
            self.recurrent_layer = nn.RNN(590, self.hidden_size, self.num_layers, batch_first=True)
        elif model_type == 1:
            self.recurrent_layer = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
       
        self.linear = nn.Linear(self.hidden_size, self.output_size)
        self.sigmoid = nn.Sigmoid()
    
        self.create_validation_measures()

    def create_validation_measures(self):

        a = "macro"  
        t = "multiclass"
        c = self.num_classes

        self.accuracy = Accuracy(task=t, num_classes=c)
        self.f1 = F1Score(task=t, average=a, num_classes=c)
        self.recall = Recall(task=t, average=a, num_classes=c)
        self.precision = Precision(task=t, average=a, num_classes=c)

    def forward(self, x):
        if self.model_type == 1:
            out, (hn, cn) = self.recurrent_layer(x)
        elif self.model_type == 0:
            out, hn = self.recurrent_layer(x)
        out = self.linear(out[:, -1, :])  # Taking the output of the last sequence step
        out = self.linear(out.T)
        return out

    def objective(self, preds, labels):
        obj = torch.nn.CrossEntropyLoss()
        return obj(preds, labels)


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def training_step(self, batch, batch_idx):
        x, y = batch
        target = y  # Assuming y is a tuple containing input and target tensors, extract the target tensor
        y_pred = self(x)
        loss = self.objective(y_pred, target)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)  # Log loss
        return loss
    

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.objective(y_pred, y)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        softmax = torch.nn.Softmax(dim=1)  # considering batch size
        preds = softmax(preds)

        preds = torch.argmax(preds, dim=1)
        measures = {"accuracy": self.accuracy, "f1": self.f1,
                    "recall": self.recall, "precision": self.precision}

        for current_key in measures.keys():
            score = measures[current_key](preds, labels)
            self.log(current_key, score, batch_size=self.batch_size,
                    on_step=True, on_epoch=True, sync_dist=True)
        
        return loss
        
