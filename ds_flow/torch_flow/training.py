

import torch.nn as nn
import torch
from tqdm import tqdm

from ds_flow.torch_flow.data_loaders import DeviceDataLoader
from ds_flow.torch_flow.evaluation import model_evaluate





def fit(epochs: int, model: nn.Module, train_loader: DeviceDataLoader, val_loader: DeviceDataLoader, optimizer, loss_fn, lr_scheduler=None, agg_func=torch.nanmean, secondary_metric=None, secondary_metric_name: str ='', save_file: str = ''):
    """Trains the model. Returns a dictionary of loss histories.
    
    Assumes that the loaders and model are already on the same device. The loader can be "on a device" by wrapping a regular DataLoader with DeviceDataLoader from nn_utils.DeviceDataLoader.


    Parameters
    ----------

    `epochs` : the number of epochs the model will train for.

    `model` : the pytorch model to train.

    `train_loader` : the DataLoader containing the training data. Should be wrapped with DeviceDataLoader so data is automatically loaded to the device.

    `val_loader` : the DataLoader containing the testing data. Should be wrapped with DeviceDataLoader so data is automatically loaded to the device.

    `optimizer` : the pytorch optimizer to train the model with.

    `loss_fn` : custom or standard pytorch loss function.

    `lr_scheduler` : (optional) a pytorch learning rate scheduler to control the learning rate.

    `agg_func` : defaults to `torch.nanmean`. The aggregation of the metrics over all of the batches for a single epoch.

    `secondary_metric` : (optional) a metric that compares predictions to targets. Could be a loss function or some kind of classification metric such as accuracy.

    `secondary_metric_name` : (optional) a string name to call the secondary metric if one is passed.

    `save_file` : (optional) default is an empty string. If a non-empty string is specified it is used to record the best model each epoch. The file should be a .pth file.

    Returns
    -------
    `dict` : dictionary of the training and testing losses. Also the secondary metrics of those are specified.
    """
    history = {
        'train_loss': [],
        'test_loss': [],
    }

    if secondary_metric != None:
        history[f'train_{secondary_metric_name}'] = []
        history[f'test_{secondary_metric_name}'] = []
    

        # Pre-training evalutation
        test_loss = model_evaluate(model, val_loader, loss_fn, evaluation_type='testing data', agg_func=agg_func)
        train_loss = model_evaluate(model, train_loader, loss_fn, evaluation_type='training data', agg_func=agg_func)

        secondary_metric_string = ''
        if secondary_metric != None:
            test_metric = model_evaluate(model, val_loader, secondary_metric, evaluation_type=f'testing data {secondary_metric_name}', agg_func=agg_func)
            train_metric = model_evaluate(model, train_loader, secondary_metric, evaluation_type=f'training data {secondary_metric_name}', agg_func=agg_func)
            history[f'train_{secondary_metric_name}'].append(train_metric)
            history[f'test_{secondary_metric_name}'].append(test_metric)
            secondary_metric_string = f"\tTrain {secondary_metric_name}: {train_metric}\tTest {secondary_metric_name}: {test_metric}"
        
        lr_string = ''
        if lr_scheduler != None:
            lr_string = f"Learning rate: {lr_scheduler.get_last_lr()[0]:.5f}\t"


        print(f"Epoch: {-1}\t{lr_string}Train Loss: {train_loss:.4f}\tTest Loss: {test_loss:.4f}{secondary_metric_string}")
        
        history['train_loss'].append(train_loss)
        history['test_loss'].append((test_loss.cpu()))




    for epoch in range(epochs):
        # Training Phase
        model.train()

        loop = tqdm(train_loader, leave=True, desc=f"Training Epoch {epoch}")
        training_losses = []
        for batch in loop:
            images, y_true = batch
            y_preds = model(images)
            loss = loss_fn(y_preds, y_true)
            training_losses.append(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            optimizer.zero_grad()

        if lr_scheduler != None:
                lr_scheduler.step()
        

        # Validation phase
        test_loss = model_evaluate(model, val_loader, loss_fn, evaluation_type='testing data', agg_func=agg_func)

        secondary_metric_string = ''
        if secondary_metric != None:
            test_metric = model_evaluate(model, val_loader, secondary_metric, evaluation_type=f'testing data {secondary_metric_name}', agg_func=agg_func)
            train_metric = model_evaluate(model, train_loader, secondary_metric, evaluation_type=f'training data {secondary_metric_name}', agg_func=agg_func)
            history[f'train_{secondary_metric_name}'].append(train_metric)
            history[f'test_{secondary_metric_name}'].append(test_metric)
            secondary_metric_string = f"\tTrain {secondary_metric_name}: {train_metric}\tTest {secondary_metric_name}: {test_metric}"
        
        lr_string = ''
        if lr_scheduler != None:
            lr_string = f"Learning rate: {lr_scheduler.get_last_lr()[0]:.5f}\t"


        print(f"Epoch: {epoch}\t{lr_string}Train Loss: {agg_func(torch.FloatTensor(training_losses)):.4f}\tTest Loss: {test_loss:.4f}{secondary_metric_string}")
        
        if epoch==0 and (save_file != ''):
             torch.save(model.state_dict(), save_file)
             print(f"Saving model state at '{save_file}'")
        elif (test_loss < min(history['test_loss'])) and (save_file != ''):
             torch.save(model.state_dict(), save_file)
             print(f"Saving model state at '{save_file}'")
        
        history['train_loss'].append(agg_func(torch.FloatTensor(training_losses)))
        history['test_loss'].append((test_loss.cpu()))


    
    
    return history