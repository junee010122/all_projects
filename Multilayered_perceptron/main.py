
import sys
import lightning as L
import torch
import pickle
import os

from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import LearningRateMonitor

from utils.models import Network
from utils.data import load_datasets
from utils.plots import plot_evaluation, plot_decision_boundary

from utils.general import load_config, log_params, clear_logfile, collect_csv
from sklearn.model_selection import KFold


torch.manual_seed(0)

def run_example(params):



    log_params(params)
    datasets = load_datasets(params)
    
    num_folds = 5

    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    
    train_dataset = datasets['train'].dataset
    predictions=[]

    for train_idx, val_idx in kf.split(train_dataset):
        
        batchsize = params["network"]["batch_size"]

        train_data_fold = torch.utils.data.Subset(train_dataset, train_idx)
        val_data_fold = torch.utils.data.Subset(train_dataset, val_idx)
        train_loader_fold = torch.utils.data.DataLoader(train_data_fold, batch_size=batchsize, shuffle=True)
        val_loader_fold = torch.utils.data.DataLoader(val_data_fold, batch_size=batchsize)

        # Create: Neural Network

        model = Network(params)

        # Create: Logger
        
        if params["network"]["type"] == 'classification':
            exp_logger = CSVLogger(save_dir=params["paths"]["results"], name='classification')
        
        if params["network"]["type"] == 'regression':
            exp_logger = CSVLogger(save_dir=params["paths"]["results"], name='regression')


        # Create: Trainer

        num_epochs = params["network"]["num_epochs"]
        strategy = params["system"]["gpus"]["strategy"]
        num_devices = params["system"]["gpus"]["num_devices"]
        accelerator = params["system"]["gpus"]["accelerator"]
        lr_monitor = LearningRateMonitor(logging_interval="epoch")

        trainer = L.Trainer(callbacks=[lr_monitor],
                        accelerator=accelerator, strategy=strategy,
                        devices=num_devices, max_epochs=num_epochs,
                        log_every_n_steps=1, logger=exp_logger)

        # Train: Model

        train, valid = datasets["train"], datasets["test"]
        trainer.fit(model=model, train_dataloaders=train, val_dataloaders=valid)
        
        with torch.no_grad():
            val_logits = torch.tensor([], dtype=torch.float32)
            for val_batch in val_loader_fold:
                val_logits = torch.cat([val_logits, model(val_batch[0])])

        predictions.append(val_logits)
    all_predictions = torch.cat(predictions, dim=0)

    file_name = 'all_predictions.pkl'
    directory = '/Users/june/Documents/results/NNclass/sanity_check/classification/LR0.005'
    file_path = os.path.join(directory, file_name)
    with open(file_path, 'wb') as f:
        pickle.dump(all_predictions, f)

    #plot_decision_boundary(Network, datasets['train'].dataset, all_predictions)
    

if __name__ == "__main__":
    # Load: YAML Parameters

    params = load_config(sys.argv)

    # Load: Launch Experiment

    run_example(params)

    print("ENDED!!") 
    
    if params["network"]["type"] == 'classification':
        path_list=collect_csv("/Users/june/Documents/results/NNclass/sanity_check/classification/")

    if params["network"]["type"] == "regression":
        path_list=collect_csv("/Users/june/Documents/results/NNclass/sanity_check/regression")
    #plot_evaluation(path_list)

