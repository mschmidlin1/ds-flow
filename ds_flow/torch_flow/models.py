from typing import Counter
import torch
import torch.nn as nn

from ds_flow.torch_flow.torch_utils import conv2d_output_size, max_pool_output_size




class ImageClassificationBase(nn.Module):
    def cross_entropy_loss(self, batch, weight=None):
        images, labels = batch
        labels = labels.type(torch.int64)
        out = self(images)             # Generate predictions
        loss = F.cross_entropy(out, labels, weight=weight) # Calculate loss
        return loss


class LogisticModel(ImageClassificationBase):

    """Feedfoward neural network with 1 hidden layer"""
    def __init__(self, in_size, hidden_size, out_size):
        super().__init__()
        # hidden layer
        self.linear1 = nn.Linear(in_size, hidden_size)
        # output layer
        self.linear2 = nn.Linear(hidden_size, out_size)
        
    def forward(self, xb):
        # Flatten the image tensors
        xb = xb.view(xb.size(0), -1)
        # Get intermediate outputs using hidden layer
        out = self.linear1(xb)
        # Apply activation function
        out = F.relu(out)
        # Get predictions using output layer
        out = self.linear2(out)
        return out
    
def conv2d_pool_block(in_channels, out_channels, kernel_size=3, padding=0, stride=1, pool_ksize=2, pool_stride=None, pool_padding=0):
    if pool_stride==None:
        pool_stride=pool_ksize
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride), 
        nn.MaxPool2d(kernel_size=pool_ksize, stride=pool_stride, padding=pool_padding)]
    return layers

class CNNModel(ImageClassificationBase):
    """CNN with arbitrary number of layers.
    Assumes that the images are square (same height as width."""
    def __init__(self, img_size, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 4, kernel_size=3)
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(4, 16, kernel_size=3)
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(30*30*16, 128)
        self.fc2 = nn.Linear(128, num_classes)



    def forward(self, xb):
            out = self.conv1(xb)
            out = self.max_pool1(out)
            out = self.conv2(out)
            out = self.max_pool2(out)
            out = out.view(out.size(0), -1)
            out = self.fc1(out)
            out = self.fc2(out)
            return out
    
class CNN2(ImageClassificationBase):
    """CNN with arbitrary number of layers.
    Assumes that the images are square (same height as width."""
    def __init__(self, img_size, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 4, kernel_size=3)
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(4, 16, kernel_size=3)
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu2 = nn.ReLU()
        self.fc1 = nn.Linear(30*30*16, 128)
        self.fc2 = nn.Linear(128, num_classes)



    def forward(self, xb):
            out = self.conv1(xb)
            out = self.max_pool1(out)
            out = self.relu1(out)
            out = self.conv2(out)
            out = self.max_pool2(out)
            out = self.relu2(out)
            out = out.view(out.size(0), -1)
            out = self.fc1(out)
            out = self.fc2(out)
            return out
    
def create_basic_cnn(num_classes, in_channels=1, img_size=32):
    """Creates a basic CNN that can handle variable input image sizes and channels.
    
    Args:
        num_classes (int): Number of output classes
        in_channels (int, optional): Number of input channels (1 for grayscale, 3 for RGB). Defaults to 1.
        img_size (int, optional): Size of input images (assumes square). Defaults to 32.
        
    Returns:
        nn.Sequential: A CNN with structure:
            Conv2d -> ReLU -> MaxPool2d -> Conv2d -> ReLU -> MaxPool2d -> Flatten -> Linear -> ReLU -> Linear
    """
    # Calculate sizes after each layer
    # First conv block
    conv1_size = conv2d_output_size(img_size, kernel_size=3)  # First conv layer
    pool1_size = max_pool_output_size(conv1_size, pool_ksize=2, pool_stride=2)  # First pool layer
    
    # Second conv block
    conv2_size = conv2d_output_size(pool1_size, kernel_size=3)  # Second conv layer
    pool2_size = max_pool_output_size(conv2_size, pool_ksize=2, pool_stride=2)  # Second pool layer
    
    # Calculate flattened feature size
    flattened_features = pool2_size * pool2_size * 32
    
    return nn.Sequential(
        nn.Conv2d(in_channels, 16, kernel_size=3),  # Now works with any number of input channels
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),  # 30x30 -> 15x15
        
        # Second conv block
        nn.Conv2d(16, 32, kernel_size=3),  # 15x15 -> 13x13
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),  # 13x13 -> 6x6
        
        # Flatten layer
        nn.Flatten(),  # 6x6x32 = 1152 features
        
        # Fully connected layers
        nn.Linear(flattened_features, 128),
        nn.ReLU(),
        nn.Linear(128, num_classes)
    )
    
