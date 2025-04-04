

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def inference(image, device, model, class_lookup):
    image = image.to(device)
    image = torch.unsqueeze(image, dim=0)
    with torch.inference_mode():
        preds = model(image)
    probs = torch.softmax(preds, dim=1)[0]
    pred_class_idx = torch.argmax(probs).int().item()
    pred_class_label = class_lookup[pred_class_idx]
    return pred_class_label, probs[pred_class_idx].float().item()


def model_evaluate(model, loader, loss_fn, agg_func=torch.nanmean, evaluation_type=''):
    """
    Assumes that the loader and model are already on the same device. The loader can be "on a device" by wrapping a regular DataLoader with DeviceDataLoader from nnutils.DeviceDataLoader.
    """
    losses = []
    model.eval()
    with torch.inference_mode():
        loop = tqdm(loader, leave=True, desc="Evaluation "+evaluation_type)
        for batch in loop:
            x, y = batch
            y_preds = model(x)
            losses.append(loss_fn(y_preds, y))
    #the nan values seem random so for now we will ignore them using torch.nanmean
    return agg_func(torch.FloatTensor(losses))

def combine_histories(*histories):
    """
    Takes in an arbitrary number of history dictionaries and combines them. 

    All dictionaries passed must have the same keys. The values for each dictionary value must be lists.
    """
    history = {key: [] for key in histories[0]}
    
    for hist in histories:
        for key in history.keys():
            history[key].extend(hist[key])
    return history

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def plot_history(history):

    history_keys = list(history.keys())
    history_keys.remove('train_loss')
    history_keys.remove('test_loss')
    secondary_train_key = history_keys[0] if 'train' in history_keys[0] else history_keys[1]
    secondary_test_key = history_keys[1] if 'test' in history_keys[1] else history_keys[0]
    secondary_metric_name = secondary_train_key.split("_")[1]


    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 3.5))

    total_epochs = len(history['train_loss'])
    epochs = list(range(total_epochs))

    ax[0].set_title("Model Loss")
    ax[0].set_ylabel("Loss")
    ax[0].set_xlabel("Epoch")
    ax[0].plot(epochs, history['train_loss'], c='blue', label='train loss')
    ax[0].plot(epochs, history['test_loss'], c='orange', label='test loss')
    ax[0].grid()
    ax[0].legend()

    ax[1].set_title(f"Model {secondary_metric_name}")
    ax[1].set_ylabel(secondary_metric_name)
    ax[1].set_xlabel("Epoch")
    ax[1].plot(epochs, history[secondary_train_key], c='blue', label=f'train {secondary_metric_name}')
    ax[1].plot(epochs, history[secondary_test_key], c='orange', label=f'test {secondary_metric_name}')
    ax[1].grid()
    ax[1].legend()

    return fig, ax

def predict_and_visualize(model, device, dataset, class_lookup, idx=None):
    if idx==None:
        idx = np.random.randint(low=0, high=len(dataset), size=1)[0]
    image, label = dataset[idx]
    plt.imshow(image[0], cmap='gray')
    pred_label, probability = inference(image, device=device, model=model, class_lookup=class_lookup)
    print(f"Index: {idx}", 'Label:', class_lookup[label], "\tPredicted:", pred_label, f"   Probability={probability:.2f}")