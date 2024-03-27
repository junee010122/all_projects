import torch
import lightning as L
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics import Accuracy, Precision, Recall, F1Score, MeanSquaredError, MeanAbsoluteError, R2Score 
from utils.data import track_weights

class Network(L.LightningModule):
 

    def __init__(self, params):


        super().__init__()

        # Load: Model Parameters

        self.alpha = params["network"]["learning_rate"]
        self.batch_size = params["network"]["batch_size"]
        self.num_epochs = params["network"]["num_epochs"]
        self.num_classes = params["datasets"]["num_classes"]
        self.problem_type = params["network"]["type"]
        self.all_predict = []

        # Define: Model Architecture

        if self.problem_type == 'classification':
        
            self.arch = torch.nn.Sequential(torch.nn.Linear(2, 128),
                                            torch.nn.ReLU(),
                                            torch.nn.Linear(128, 64),
                                            torch.nn.ReLU(),
                                            torch.nn.Linear(64, 32),
                                            torch.nn.ReLU(),
                                            torch.nn.Linear(32,16),
                                            torch.nn.ReLU(),
                                            torch.nn.Linear(16,2),
                                            torch.nn.ReLU()
                                            )

        if self.problem_type == 'regression':

            self.arch = torch.nn.Sequential(torch.nn.Linear(1, 64),
                                            torch.nn.ReLU(),
                                            torch.nn.Linear(64, 32),
                                            torch.nn.ReLU(),
                                            torch.nn.Linear(32,16),
                                            torch.nn.ReLU(),
                                            torch.nn.Linear(16,1))

        model = self.arch

        self.weights = track_weights(model) 

        # Create: Validation Measures

        self.create_validation_measures()

    def create_validation_measures(self):

        if self.problem_type == 'classification':
            a = "macro"  
            t = "multiclass"
            c = self.num_classes

            self.accuracy = Accuracy(task=t, num_classes=c)
            self.f1 = F1Score(task=t, average=a, num_classes=c)
            self.recall = Recall(task=t, average=a, num_classes=c)
            self.precision = Precision(task=t, average=a, num_classes=c)

        if self.problem_type == 'regression':
            
            self.mse = MeanSquaredError()
            self.mae = MeanAbsoluteError()
            self.r_squared = R2Score()


    def configure_optimizers(self):

        # Create: Optimzation Routine

        #if classification turn this on
        optimizer = torch.optim.SGD(self.parameters(),
                                    lr=self.alpha, momentum=0.9)

        #optimizer = torch.optim.Adam(self.parameters(), lr=self.alpha)

        # Create: Learning Rate Schedular

        #lr_scheduler = CosineAnnealingLR(optimizer, T_max=self.num_epochs)
        #lr_scheduler_config = {"scheduler": lr_scheduler,
                               #"interval": "epoch", "frequency": 1}

        #return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}
        return {"optimizer": optimizer}

    def objective(self, preds, labels):
        
        if self.problem_type == 'classification':
            obj = torch.nn.CrossEntropyLoss()
            
            return obj(preds, labels)
        if self.problem_type == 'regression':
            obj = torch.nn.MSELoss()

            return obj(preds, labels)

    def training_step(self, batch, batch_idx):


        samples, labels = batch

        # turn this on when regression problem
        #labels = labels.unsqueeze(1)

        # Gather: Predictions
        
        preds = self.arch(samples)

        # Calculate: Objective Loss

        loss = self.objective(preds, labels)

        self.log("train_error", loss, batch_size=self.batch_size,
                 on_step=True, on_epoch=True, sync_dist=True)
        
        return loss

    def validation_step(self, batch, batch_idx):

        #from IPython import embed
        #embed()
        samples, labels = batch

        # Gather: Predictions

        preds = self.arch(samples)
        

        # Calculate: Objective Loss

        # for regression problem turn this on
        #labels=labels.unsqueeze(1)
        
        loss = self.objective(preds, labels)
        

        self.log("valid_error", loss, batch_size=self.batch_size,
                 on_step=True, on_epoch=True, sync_dist=True)

        # Calculate: Confusion Matrix Analytics

        if self.problem_type == 'classification':

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
        
        if self.problem_type == 'regression':
            
            measures = {"MSE":self.mse, "MAE":self.mae, "R2":self.r_squared}
            for current_key in measures.keys():
                score = measures[current_key](preds, labels)
                self.log(current_key, score, batch_size=self.batch_size,
                         on_step=True, on_epoch=True, sync_dist=True)

            return loss

        
    def forward(self, samples):
        return self.arch(samples)
