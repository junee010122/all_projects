
import torch
import torch.nn as nn
import lightning as L

from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics import Accuracy, Precision, Recall, F1Score


class Network(L.LightningModule):

    def __init__(self, params):

        super().__init__()

        self.data_id = params["dataset"]["type"]
        self.arch_id = params["network"]["arch"]
        self.opti = params["network"]["optimizer"]
        self.num_epochs = params["network"]["num_epochs"]
        self.batch_size = params["network"]["batch_size"]
        self.learning_rate = params["network"]["learning_rate"]

        self.num_preds = params["dataset"]["num_predicted"]

        # Set: Input Image Dimensions based on dataset
        # - 0 = MNIST, 1 = CIFAR

        if self.data_id == 0:
            self.in_dims = 1
        else:
            self.in_dims = 3

        # Select: Network Architecture

        self.select_architecture()

        # Create: Validation Measures

        self.create_validation_measures()

    def create_validation_measures(self):
        """
        Creates confusion matrix measures for validation assessment
        """

        a = "macro"
        t = "multiclass"
        c = self.num_preds

        self.accuracy = Accuracy(task=t, num_classes=c)
        self.f1 = F1Score(task=t, average=a, num_classes=c)
        self.recall = Recall(task=t, average=a, num_classes=c)
        self.precision = Precision(task=t, average=a, num_classes=c)

    def select_architecture(self):
        """
        Load the DL model architecture
        """

        if self.arch_id == 0:

            # Create: Feature Extraction

            self.extract = nn.Sequential(
                                         # Dims: [C, H, W]

                                         nn.Conv2d(in_channels=self.in_dims,
                                                   out_channels=64, stride=1,
                                                   kernel_size=3, padding=1),

                                         # Dims: [64, H, W]

                                         nn.ReLU(),
                                         nn.MaxPool2d(2, 2),

                                         # Dims: [64, H / 2, W / 2]

                                         nn.Conv2d(in_channels=64,
                                                   out_channels=128, stride=1,
                                                   kernel_size=3, padding=1),
                                         nn.ReLU(),

                                         # Dims: [128, H / 2, W / 2]

                                         nn.MaxPool2d(2, 2))

                                         # Dims: [128, H / 4, W / 4]

            # Create: MLP
            # - 0 = MNIST, 1 = CIFAR

            if self.data_id == 0:
                size = 128 * 7 * 7
            elif self.data_id == 1:
                size = 128 * 8 * 8

            self.predict = nn.Linear(size, self.num_preds)

        elif self.arch_id == 1:
            pass

        elif self.arch_id == 2:
            pass
        else:

            raise NotImplementedError

    def configure_optimizers(self):

        if self.opti == 0:
            optimizer = torch.optim.SGD(self.parameters(),
                                        lr=self.learning_rate, momentum=0.9)

        elif self.opti == 1:
            optimizer = torch.optim.Adam(self.parameters(),
                                         lr=self.learning_rate)
        else:
            raise NotImplementedError

        lr_scheduler = CosineAnnealingLR(optimizer, T_max=self.num_epochs)
        lr_scheduler_config = {"scheduler": lr_scheduler,
                               "interval": "epoch", "frequency": 1}

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}

    def objective(self, preds, labels):

        obj = torch.nn.CrossEntropyLoss()

        return obj(preds, labels)

    def training_step(self, batch, batch_idx):

        samples, labels = batch

        # Gather: Predictions

        preds = self(samples)

        # Calculate: Objective Loss

        loss = self.objective(preds, labels)

        self.log("train_error", loss, batch_size=self.batch_size,
                 on_step=True, on_epoch=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):

        samples, labels = batch

        # Gather: Predictions

        preds = self(samples)

        # Calculate: Objective Loss

        loss = self.objective(preds, labels)

        self.log("valid_error", loss, batch_size=self.batch_size,
                 on_step=True, on_epoch=True, sync_dist=True)

        # Calculate: Confusion Matrix Analytics

        preds = torch.argmax(preds, dim=1).to(torch.int32)

        measures = {"accuracy": self.accuracy, "f1": self.f1,
                    "recall": self.recall, "precision": self.precision}

        for current_key in measures.keys():
            score = measures[current_key](preds, labels)
            self.log(current_key, score, batch_size=self.batch_size,
                     on_step=True, on_epoch=True, sync_dist=True)

    def forward(self, x):

        # Run: Feature Extraction
        # - Image --> Convolutional Feature Maps

        x = self.extract(x)

        # Run: MLP Prediction

        x = x.view(x.size()[0], -1)

        return self.predict(x)
