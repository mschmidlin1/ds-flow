
from typing import Counter
import torch
import numpy as np
import torchvision
import math

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


def conv2d_output_size(img_size, kernel_size=3 , stride=1, padding=0, dilation=1):
    return math.floor((img_size+2*padding-dilation*(kernel_size-1)-1)/stride) + 1

def max_pool_output_size(img_size, pool_ksize=2, pool_stride=2, pool_padding=0, dilation=1):
    return math.floor((img_size+(2*pool_padding)-(dilation*pool_ksize-1)-1)/pool_stride) + 1

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

def ClassificationTransform(img_size=(128, 128)):
    return torchvision.transforms.Compose([
                        torchvision.transforms.Grayscale(num_output_channels=1),
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Resize(img_size),
                        torchvision.transforms.RandomCrop(img_size, padding=4, padding_mode='reflect'), 
                        torchvision.transforms.RandomHorizontalFlip(),
                        torchvision.transforms.RandomVerticalFlip(), 
                        torchvision.transforms.RandomRotation(degrees=(0, 180))
                        ])



