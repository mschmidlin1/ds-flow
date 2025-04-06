from typing import Counter
import torch
import numpy as np

import math

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def get_available_gpus() -> list:
    available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
    return available_gpus

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


def conv2d_output_size(img_size, kernel_size=3 , stride=1, padding=0, dilation=1):
    """Calculate the output size of a convolutional layer.

    This function implements the formula for computing the output size of a 2D convolution operation
    based on the input parameters according to PyTorch's conventions.

    Args:
        img_size (int): Size of the input feature map (assuming square input)
        kernel_size (int, optional): Size of the convolving kernel. Defaults to 3.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Zero-padding added to both sides of the input. Defaults to 0.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.

    Returns:
        int: Output size of the feature map after convolution

    Raises:
        ValueError: If the calculated output size would be less than 1

    Example:
        >>> conv2d_output_size(32, kernel_size=3, stride=1, padding=1)
        32
    """
    output_size = math.floor((img_size+2*padding-dilation*(kernel_size-1)-1)/stride) + 1
    if output_size < 1:
        raise ValueError(f"Invalid parameters: output size {output_size} would be less than 1. Input size {img_size} is too small for given parameters.")
    return output_size

def max_pool_output_size(img_size, pool_ksize=2, pool_stride=2, pool_padding=0, dilation=1):
    """Calculate the output size of a max pooling layer.

    This function implements the formula for computing the output size of a 2D max pooling operation
    based on the input parameters according to PyTorch's conventions.

    Args:
        img_size (int): Size of the input feature map (assuming square input)
        pool_ksize (int, optional): Size of the pooling window. Defaults to 2.
        pool_stride (int, optional): Stride of the pooling operation. Defaults to 2.
        pool_padding (int, optional): Zero-padding added to both sides of the input. Defaults to 0.
        dilation (int, optional): Spacing between pooling window elements. Defaults to 1.

    Returns:
        int: Output size of the feature map after max pooling

    Raises:
        ValueError: If the calculated output size would be less than 1

    Example:
        >>> max_pool_output_size(32, pool_ksize=2, pool_stride=2)
        16
    """
    output_size = math.floor((img_size+(2*pool_padding)-(dilation*pool_ksize-1)-1)/pool_stride) + 1
    if output_size < 1:
        raise ValueError(f"Invalid parameters: output size {output_size} would be less than 1. Input size {img_size} is too small for given parameters.")
    return output_size

def count_model_parameters(model):
    params_list = []
    for l in list(model.parameters()):
        params_list.append(torch.prod(torch.tensor(l.shape)))
    return torch.sum(torch.tensor(params_list))


def parse_model(model, input_dims: list):

    layer_lookup = {
        'Conv2d': conv2d_output_size,
        'MaxPool2d': max_pool_output_size,
    }

    layer_funcs = []

    for layer in str(model).split("\n")[1:-1]:
        layer_type = layer.split(": ")[1]
        layer_name = layer_type.split("(")[0]
        if layer_name in layer_lookup.keys():
            layer_funcs.append(layer_lookup[layer_name])


    for dim in input_dims:
        current_dim = dim
        for func in layer_funcs:
            current_dim = func(current_dim)

        print(f"Dimension {dim} -> {current_dim}")



def convert_bit_depth(image, target_bit_depth=np.uint8):
    current_max_value = np.iinfo(image.dtype).max
    target_max_value = np.iinfo(target_bit_depth).max
    converted_image = image * (target_max_value/current_max_value)

    converted_image = np.round(converted_image).astype(target_bit_depth)

    return converted_image

class Squeeze(object):
    def __init__(self, dim=None):
        self.dim = dim

    def __call__(self, img):
        img = torch.squeeze(img)
        return img









import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

def prep_RNN_data(bars_df, cols=['close'], test_pct=0.25, scaler=None):
    array = bars_df[cols].values
    if scaler!=None:
        array = scaler.fit_transform(array)
    train_size = len(array) - int(len(array)*test_pct)
    train_array = array[:train_size, :]
    test_array = array[train_size:, :]
    return train_array, test_array

class RNNDataset(torch.utils.data.Dataset):
    def __init__(self, data, seq_len, pred_idx=0):
        """
        Takes in `data` which is a numpy array and `seq_len` which is an int.

        `pred_idx` is the column index we want to predict.
        """
        self.data = data
        self.seq_len = seq_len
        self.pred_idx = pred_idx
        self.data = torch.from_numpy(data).float()

    def __len__(self):
        return (len(self.data) - self.seq_len) - 1
    def __getitem__(self, index):
        return self.data[index : index+self.seq_len, :], self.data[index+self.seq_len, self.pred_idx:self.pred_idx+1]

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)
    
class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(RNN, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.num_layers = num_layers

        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):

        out, hn = self.rnn(x) #by not providing hidden state it is automatically set to 0's

        # out.size() --> batch_size, sequence_length, hidden_size --> torch.Size([32, 50, 100])
        # out[:, -1, :] --> batch_size, hidden_size -->torch.Size([32, 100]) --> just want last time step! 
        out = self.fc(out[:, -1, :]) 
        # out.size() --> batch_size, output_dim --> 100, 1
        return out

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.num_layers = num_layers

        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):

        out, (hn, cn) = self.lstm(x) #by not providing hidden state and cell state they are automatically set to 0's

        # out.size() --> batch_size, sequence_length, hidden_size --> torch.Size([32, 50, 100])
        # out[:, -1, :] --> batch_size, hidden_size -->torch.Size([32, 100]) --> just want last time step! 
        out = self.fc(out[:, -1, :]) 
        # out.size() --> batch_size, output_dim --> 100, 1
        return out
    
class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(GRU, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.num_layers = num_layers

        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):

        out, hn = self.gru(x) #by not providing hidden state it is automatically set to 0's

        # out.size() --> batch_size, sequence_length, hidden_size --> torch.Size([32, 50, 100])
        # out[:, -1, :] --> batch_size, hidden_size -->torch.Size([32, 100]) --> just want last time step! 
        out = self.fc(out[:, -1, :]) 
        # out.size() --> batch_size, output_dim --> 100, 1
        return out

def fit_model(num_epochs: int, model: nn.Module, train_loader: torch.utils.data.DataLoader, test_loader: torch.utils.data.DataLoader, loss_fn, optimizer: torch.optim, print_divider=10):
    training_losses = []
    testing_losses = []
    
    for e in range(num_epochs):
        # Forward pass
        train_losses = []
        model.train()
        for batch in train_loader:
            X_train, y_train = batch
            y_train_pred = model(X_train)
            loss = loss_fn(y_train_pred, y_train)
            train_losses.append(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        test_losses = []
        with torch.inference_mode():
            model.eval()
            for batch in test_loader:
                X_test, y_test = batch
                y_test_pred = model(X_test)
                loss = loss_fn(y_test_pred, y_test)
                test_losses.append(loss)

        training_losses.append(torch.mean(torch.FloatTensor(train_losses)))
        testing_losses.append(torch.mean(torch.FloatTensor(test_losses)))
        if e % print_divider == 0:
            print(f"Epoch {e}  \tTrain MSE: {torch.mean(torch.FloatTensor(train_losses)):.5f}\tTest MSE: {torch.mean(torch.FloatTensor(test_losses)):.5f}")
    return training_losses, testing_losses

def plot_losses(num_epochs, training_losses, testing_losses, title_addition=''):
    training_losses = torch.FloatTensor(training_losses).cpu().numpy()
    testing_losses = torch.FloatTensor(testing_losses).cpu().numpy()

    fig, ax = plt.subplots()
    ax.plot(np.array(range(num_epochs)), training_losses, c='blue', label='Test Loss')
    ax.plot(np.array(range(num_epochs)), testing_losses, c='orange', label='Train Loss')
    ax.legend()
    ax.set_ylabel("Error")
    ax.set_xlabel("Epochs")

    ax.set_title(f"Training Vs Testing Losses\n{title_addition}")

def rnn_type_predict(model: nn.Module, array: torch.Tensor, device):
    """
    Predicts the next value in a sequence of values. 

    Parameters:
    -----------
    `model` : The pytorch model to predict the next value in the sequence. 
    The model must be Recurrent in nature. Most likely meaning either RNN, LSTM, or GRU.
    `array` : The sequence of values to find the next value of.
    `device` : Either a string or pytorch device.
    """
    model.eval()
    array = torch.unsqueeze(array, 0) #unsqueeze once for batch size of 1
    array = array.to(device)
    with torch.inference_mode():
        pred = model(array)

    return float(pred.squeeze().cpu())

def visualize_predictions(model, historical_array, test_dataset, device, title=''):

    #### Make predictions
    y_pred = []
    y_true = []
    for i in range(len(test_dataset)):
        x_test, y_test = test_dataset[i]
        y_pred.append(rnn_type_predict(model, x_test, device))
        y_true.append(float(y_test.squeeze().cpu()))


    #### Check classification
    actual_rise = []
    pred_rise = []
    for i in range(1, len(y_pred)):
        if y_pred[i] > y_true[i-1]:
            pred_rise.append(1)
        else:
            pred_rise.append(0)
        
        if y_true[i] > y_true[i-1]:
            actual_rise.append(1)
        else:
            actual_rise.append(0)
    print(classification_report(actual_rise, pred_rise))



    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    x = np.arange((len(historical_array)+len(y_true)))
    y = np.append(historical_array, y_true)

    x_preds = np.arange(start=len(historical_array), stop=len(y))

    #print(preds.shape, x_preds.shape)

    fig, ax = plt.subplots()
    ax.plot(x, y, label='prices')
    ax.plot(x_preds, y_pred, c='orange', label='predictions')
    ax.set_title(title)
    ax.set_ylabel("Closing Price Normalized")
    ax.set_xlabel("Days from 2015 to 2023")
    ax.legend()
    plt.show()
    return ax

def visualize_chained_predictions(model, train_array, test_array, device, title=''):
    x = np.arange((len(train_array)+len(test_array)))

    y = np.append(np.squeeze(train_array), np.squeeze(test_array))
    

    preds = test_array[:51]
    for i in range(50, len(test_array)-1):
        X = torch.from_numpy(preds[i-50:i])
        X = X.type(torch.float)
        pred = rnn_type_predict(model, X, device)
        preds = np.expand_dims(np.append(preds, np.array(pred).reshape(1)), 1)

    

    preds = np.squeeze(preds[51:])
    x_preds = np.arange(start=len(train_array)+50, stop=len(y)-1)

    #print(preds.shape, x_preds.shape)

    fig, ax = plt.subplots()

    ax.plot(x, y, label='prices')
    ax.plot(x_preds, preds, c='orange', label='predictions')
    ax.set_title(title)
    ax.set_ylabel("Closing Prices Normalized")
    ax.set_xlabel("Days from 2015 to 2023")
    ax.legend()
    plt.show()
    return ax