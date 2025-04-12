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
    

class CIFAR10Net(nn.Module):
    """A convolutional neural network architecture optimized for CIFAR-10 classification.
    
    This network is inspired by VGG-style architectures but adapted for small images.
    It uses a series of convolutional blocks with increasing filter counts (64->128->256)
    followed by a classifier head. The network can handle arbitrary input sizes through
    dynamic computation of feature dimensions.

    Architecture Details:
    -------------------
    The network consists of three main feature blocks followed by a classifier:

    Feature Extraction Blocks:
    1. First Block (input_size -> input_size/2):
       - Two Conv2d(in_channels->64->64) layers with batch normalization
       - Maintains spatial dimensions using padding=1
       - Ends with MaxPool2d reducing spatial size by 2
       - Purpose: Initial feature extraction, edge detection, simple patterns

    2. Second Block (input_size/2 -> input_size/4):
       - Two Conv2d(64->128->128) layers with batch normalization
       - Maintains spatial dimensions using padding=1
       - Ends with MaxPool2d reducing spatial size by 2
       - Purpose: Medium complexity feature detection (textures, patterns)

    3. Third Block (input_size/4 -> input_size/8):
       - Two Conv2d(128->256->256) layers with batch normalization
       - Maintains spatial dimensions using padding=1
       - Ends with MaxPool2d reducing spatial size by 2
       - Purpose: High-level feature detection (complex shapes, object parts)

    Classifier Head:
    - Flattens final feature maps
    - Dense layer reducing to 512 features with ReLU
    - Dropout(0.5) for regularization
    - Final classification layer to num_classes

    Key Design Choices:
    -----------------
    1. Batch Normalization:
       - Added after each conv layer
       - Stabilizes training and allows higher learning rates
       - Adds some regularization effect
       
    2. Progressive Filter Growth:
       - Starts with 64 filters, doubles at each block (64->128->256)
       - Compensates for decreasing spatial dimensions
       - Allows learning of more complex features as receptive field grows
       
    3. Padding Strategy:
       - Uses padding=1 in all conv layers
       - Maintains spatial dimensions within blocks
       - Only reduces dimensions through max pooling
       
    4. Dropout:
       - Applied only in classifier head
       - Rate of 0.5 (drops 50% of connections)
       - Prevents co-adaptation of features and reduces overfitting
       
    5. Weight Initialization:
       - Kaiming initialization for conv layers (considering ReLU)
       - Normal initialization for linear layers
       - Proper initialization helps with training dynamics

    Training Recommendations:
    ----------------------
    - Epochs: 30-50 minimum
    - Optimizer: SGD with momentum (0.9) and weight decay (5e-4)
    - Learning Rate: Start at 0.1 with cosine annealing
    - Batch Size: 128 or 256 if memory allows

    Args:
        num_classes (int): Number of output classes (default: 10 for CIFAR-10)
        in_channels (int): Number of input channels (1 for grayscale, 3 for RGB)
        img_size (int): Size of input images (assumes square input)

    Input:
        - x: tensor of shape (batch_size, in_channels, img_size, img_size)
        
    Output:
        - tensor of shape (batch_size, num_classes)
    """
    
    def __init__(self, num_classes=10, in_channels=3, img_size=32):
        super().__init__()
        
        # Calculate sizes after each layer
        # First block
        conv1_1_size = conv2d_output_size(img_size, kernel_size=3, padding=1)  # Same size due to padding
        conv1_2_size = conv2d_output_size(conv1_1_size, kernel_size=3, padding=1)  # Same size due to padding
        pool1_size = max_pool_output_size(conv1_2_size, pool_ksize=2, pool_stride=2)  # Halve the size
        
        # Second block
        conv2_1_size = conv2d_output_size(pool1_size, kernel_size=3, padding=1)  # Same size due to padding
        conv2_2_size = conv2d_output_size(conv2_1_size, kernel_size=3, padding=1)  # Same size due to padding
        pool2_size = max_pool_output_size(conv2_2_size, pool_ksize=2, pool_stride=2)  # Halve the size
        
        # Third block
        conv3_1_size = conv2d_output_size(pool2_size, kernel_size=3, padding=1)  # Same size due to padding
        conv3_2_size = conv2d_output_size(conv3_1_size, kernel_size=3, padding=1)  # Same size due to padding
        pool3_size = max_pool_output_size(conv3_2_size, pool_ksize=2, pool_stride=2)  # Halve the size
        
        # Calculate final flattened size
        self.flattened_size = pool3_size * pool3_size * 256
        
        self.features = nn.Sequential(
            # First block - input: img_size x img_size x in_channels
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Second block - input: (img_size/2) x (img_size/2) x 64
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Third block - input: (img_size/4) x (img_size/4) x 128
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # output: (img_size/8) x (img_size/8) x 256
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flattened_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def forward(self, x):
        """Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, img_size, img_size)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes)
        """
        x = self.features(x)
        x = self.classifier(x)
        return x
    
    def _initialize_weights(self):
        """Initialize model weights using Kaiming/Normal initialization.
        
        Conv layers use Kaiming initialization with ReLU adjustment.
        Batch norm layers are initialized with weight=1, bias=0.
        Linear layers use normal initialization with std=0.01.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

class AlexNet(nn.Module):
    """AlexNet: The architecture that started the deep learning revolution in computer vision.
    
    Historical Significance:
    ---------------------
    AlexNet, introduced by Krizhevsky et al. in 2012, marked a pivotal moment in computer vision
    by winning the ImageNet Large Scale Visual Recognition Challenge (ILSVRC) with a TOP-5 error 
    rate of 15.3%, compared to 26.2% achieved by the second-best entry. This breakthrough 
    demonstrated the power of deep convolutional neural networks and GPU-accelerated computing
    in computer vision tasks.

    Key Innovations:
    --------------
    1. Scale: Much deeper and wider than previous CNNs (60 million parameters)
    2. ReLU Activation: First use of ReLU instead of tanh, enabling faster training
    3. Local Response Normalization (LRN): Aids generalization (though later replaced by BatchNorm)
    4. Overlapping Pooling: Reduced overfitting compared to non-overlapping pooling
    5. Dropout: Novel regularization technique to prevent co-adaptation of features
    6. Data Augmentation: Introduced image translations, horizontal reflections, and color PCA noise
    7. GPU Implementation: First CNN trained on GPUs (2 GTX 580s)

    Architecture Details:
    ------------------
    - 5 convolutional layers with decreasing kernel sizes (11,5,3)
    - 3 fully connected layers
    - ReLU activations after every conv and fc layer
    - Local Response Normalization after first two conv layers
    - Max pooling layers with overlapping windows
    - Dropout (0.5) in fully connected layers
    
    Deviations from Original:
    ----------------------
    1. Input Size: Original was fixed at 224x224, this implementation allows variable size
    2. LRN Layers: Optional as they're rarely used today (BatchNorm is preferred)
    3. GPU Split: Original split computation across 2 GPUs, this version runs on any device
    4. Number of Classes: Original had 1000 classes, this allows any number

    Training Recommendations:
    ----------------------
    - Optimizer: SGD with momentum (0.9)
    - Learning Rate: Start at 0.01, reduce by 10x when validation error plateaus
    - Weight Decay: 0.0005
    - Batch Size: 128 (original used 128 split across 2 GPUs)
    - Data Augmentation: Random crops, horizontal flips, PCA color augmentation

    Args:
        num_classes (int): Number of output classes
        in_channels (int): Number of input channels (default: 3 for RGB)
        img_size (int): Size of input images (assumes square input)
        use_lrn (bool): Whether to use Local Response Normalization (default: False)
        
    Input:
        - x: tensor of shape (batch_size, in_channels, img_size, img_size)
        
    Output:
        - tensor of shape (batch_size, num_classes)
    """
    
    def __init__(self, num_classes, in_channels=3, img_size=224, use_lrn=False):
        super().__init__()
        
        # Calculate sizes after each layer
        # First conv block
        conv1_size = conv2d_output_size(img_size, kernel_size=11, stride=4)  # First conv layer
        pool1_size = max_pool_output_size(conv1_size, pool_ksize=3, pool_stride=2)  # First pool layer
        
        # Second conv block
        conv2_size = conv2d_output_size(pool1_size, kernel_size=5, padding=2)  # Second conv layer
        pool2_size = max_pool_output_size(conv2_size, pool_ksize=3, pool_stride=2)  # Second pool layer
        
        # Third conv block (no pooling)
        conv3_size = conv2d_output_size(pool2_size, kernel_size=3, padding=1)
        
        # Fourth conv block (no pooling)
        conv4_size = conv2d_output_size(conv3_size, kernel_size=3, padding=1)
        
        # Fifth conv block
        conv5_size = conv2d_output_size(conv4_size, kernel_size=3, padding=1)
        pool5_size = max_pool_output_size(conv5_size, pool_ksize=3, pool_stride=2)
        
        # Calculate flattened size for first fc layer
        self.flattened_size = pool5_size * pool5_size * 256
        
        # Feature layers
        layers = [
            # First block - input: img_size x img_size x in_channels
            nn.Conv2d(in_channels, 96, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),
        ]
        
        if use_lrn:
            layers.append(nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2))
            
        layers.extend([
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Second block
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
        ])
        
        if use_lrn:
            layers.append(nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2))
            
        layers.extend([
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Third block
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Fourth block
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Fifth block
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        ])
        
        self.features = nn.Sequential(*layers)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(self.flattened_size, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def forward(self, x):
        """Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, img_size, img_size)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes)
        """
        x = self.features(x)
        x = self.classifier(x)
        return x
    
    def _initialize_weights(self):
        """Initialize model weights.
        
        Uses the initialization scheme from the original paper:
        - Conv layers: Gaussian with std=0.01
        - Bias in conv2, 4, 5: constant 1 (others 0)
        - FC layers: Gaussian with std=0.005
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.005)
                nn.init.constant_(m.bias, 1)  # Initialize all FC biases to 1
                
        # Special bias initialization for certain conv layers as per paper
        for i, layer in enumerate(self.features):
            if isinstance(layer, nn.Conv2d) and i in [1, 4, 10]:  # 2nd, 4th, and 5th conv layers
                nn.init.constant_(layer.bias, 1)

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
    
    def get_optimizer(self):
        """Get the best optimizer for the model"""
        return torch.optim.SGD(
            self.parameters(),
            lr=0.1,
            momentum=0.9,
            weight_decay=5e-4
        )
    def get_loss_fn(self, class_weights=None):
        """Get the best loss function for the model"""
        if class_weights is not None:
            return nn.CrossEntropyLoss(weight=class_weights)
        else:
            return nn.CrossEntropyLoss()
    def get_lr_scheduler(self, optimizer, epochs):
        """Get the best learning rate scheduler for the model"""
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs
        )


class VGGNet(nn.Module):
    """VGGNet: A deep CNN architecture that demonstrated the power of architectural simplicity and depth.
    
    Historical Significance:
    ---------------------
    VGGNet, introduced by Simonyan & Zisserman in 2014, secured the first and second places in the 
    localization and classification tracks respectively in ILSVRC 2014. Its major contribution was 
    demonstrating that depth is crucial for good performance, and that deep networks with small 
    filters (3x3) are more effective than shallow networks with larger filters.

    Key Innovations:
    --------------
    1. Architectural Simplicity: Used only 3x3 convolutions and 2x2 max pooling throughout
    2. Increased Depth: Successfully trained networks of 16-19 layers (previous nets were much shallower)
    3. Small Filters: Showed that stacking small filters is better than using larger ones
       - Three 3x3 layers have same receptive field as one 7x7 layer
       - But with fewer parameters and more non-linearities
    4. Configuration Study: Systematically evaluated networks of different depths (11-19 layers)
    5. Multi-Scale Training: Introduced training on multiple scales for better generalization

    Architecture Details:
    ------------------
    - Multiple configurations (A-E) with increasing depth
    - All convolutions are 3x3 with stride 1 and padding 1
    - All max pooling layers are 2x2 with stride 2
    - Three fully connected layers (4096-4096-num_classes)
    - All hidden layers use ReLU activation
    
    Configurations:
    -------------
    This implementation supports different VGG configurations:
    - VGG11 (Config A): 8 conv layers
    - VGG13 (Config B): 10 conv layers
    - VGG16 (Config D): 13 conv layers
    - VGG19 (Config E): 16 conv layers
    
    The most commonly used variants are VGG16 and VGG19.

    Deviations from Original:
    ----------------------
    1. Input Size: Original was fixed at 224x224, this implementation allows variable size
    2. Batch Normalization: Optional addition (not in original paper)
    3. Number of Classes: Original had 1000 classes, this allows any number

    Training Recommendations:
    ----------------------
    - Optimizer: SGD with momentum (0.9)
    - Learning Rate: Start at 0.01, divide by 10 when validation error plateaus
    - Weight Decay: 5e-4
    - Batch Size: 256
    - Data Augmentation: Random crops, horizontal flips, color jittering
    - Multi-Scale Training: Train on different scales (256-512)

    Args:
        num_classes (int): Number of output classes
        in_channels (int): Number of input channels (default: 3 for RGB)
        img_size (int): Size of input images (assumes square input)
        config (str): VGG configuration to use ('A'=11, 'B'=13, 'D'=16, 'E'=19)
        batch_norm (bool): Whether to use batch normalization (default: False)
        
    Input:
        - x: tensor of shape (batch_size, in_channels, img_size, img_size)
        
    Output:
        - tensor of shape (batch_size, num_classes)
    """
    
    # VGG configuration variants
    # Numbers represent output channels, 'M' represents max pooling
    configs = {
        'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],  # VGG11
        'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],  # VGG13
        'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],  # VGG16
        'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']  # VGG19
    }
    
    def __init__(self, num_classes, in_channels=3, img_size=224, config='VGG16', batch_norm=False):
        super().__init__()
        
        if config not in self.configs:
            raise ValueError(f"Config must be one of {list(self.configs.keys())}")
            
        # Build feature layers based on config
        self.features = self._make_layers(self.configs[config], in_channels, batch_norm)
        
        # Calculate size after feature extraction
        current_size = img_size
        for layer in self.features:
            if isinstance(layer, nn.Conv2d):
                current_size = conv2d_output_size(current_size, kernel_size=3, padding=1)
            elif isinstance(layer, nn.MaxPool2d):
                current_size = max_pool_output_size(current_size, pool_ksize=2, pool_stride=2)
        
        # Calculate flattened size
        self.flattened_size = current_size * current_size * 512
        
        # Classifier layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flattened_size, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _make_layers(self, cfg, in_channels, batch_norm):
        """Builds the feature extraction layers based on the configuration."""
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, img_size, img_size)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes)
        """
        x = self.features(x)
        x = self.classifier(x)
        return x
    
    def _initialize_weights(self):
        """Initialize model weights using the scheme from the paper.
        
        Conv layers are initialized from a normal distribution with mean=0, std=0.01.
        Biases in conv and fc layers are initialized to 0, except for the second and
        fourth fc layers which are initialized to 1.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, 0)

    def get_optimizer(self):
        """Get the best optimizer for the model"""
        return torch.optim.SGD(
            self.parameters(),
            lr=0.1,
            momentum=0.9,
            weight_decay=5e-4
        )
    def get_loss_fn(self, class_weights=None):
        """Get the best loss function for the model"""
        if class_weights is not None:
            return nn.CrossEntropyLoss(weight=class_weights)
        else:
            return nn.CrossEntropyLoss()
    def get_lr_scheduler(self, optimizer, epochs):
        """Get the best learning rate scheduler for the model"""
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs
        )

class ResidualBlock(nn.Module):
    """Basic building block for ResNet architecture.
    
    Implements the residual learning function F(x) + x, where F(x) is the residual mapping
    to be learned and x is the identity shortcut connection.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        stride (int): Stride for first conv layer (default: 1)
        downsample (nn.Module): Optional downsampling layer for shortcut connection
    """
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        
        return out

class Bottleneck(nn.Module):
    """Bottleneck block used in ResNet-50 and deeper variants.
    
    Uses a bottleneck design with 1x1 convolutions to reduce and then expand dimensions,
    making the network more computationally efficient.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels (before expansion)
        stride (int): Stride for first conv layer (default: 1)
        downsample (nn.Module): Optional downsampling layer for shortcut connection
    """
    expansion = 4
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion,
                              kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        
        return out

class ResNet(nn.Module):
    """ResNet: Deep Residual Learning for Image Recognition.
    
    Historical Significance:
    ---------------------
    ResNet, introduced by He et al. in 2015, revolutionized deep learning by solving the
    degradation problem in very deep networks through the introduction of residual connections.
    It won the ILSVRC 2015 classification task with a top-5 error rate of 3.57%, surpassing
    human-level performance (5.1%) for the first time.

    Key Innovations:
    --------------
    1. Residual Learning: Instead of learning direct mappings, learn residual functions F(x) + x
       - Easier optimization: Identity mappings provide a clear learning objective
       - Better gradient flow: Direct connections allow gradients to flow backwards
       - Enables training of much deeper networks (up to 1000+ layers)
       
    2. Bottleneck Architecture: Efficient design for deeper networks
       - 1x1 convolutions reduce and restore dimensions
       - Significantly fewer parameters while maintaining performance
       
    3. Batch Normalization: Used extensively throughout the network
       - Enables higher learning rates
       - Reduces dependence on initialization
       - Acts as regularization
       
    4. Deep Supervision: Demonstrated that deeper networks can converge
       - Previous architectures suffered from vanishing gradients
       - ResNet showed depth can significantly improve accuracy

    Architecture Details:
    ------------------
    - Initial 7x7 conv layer with stride 2
    - 3x3 max pooling with stride 2
    - Four stages of residual blocks
    - Global average pooling
    - Final fully connected layer
    
    Variants:
    --------
    - ResNet-18: Basic blocks [2,2,2,2]
    - ResNet-34: Basic blocks [3,4,6,3]
    - ResNet-50: Bottleneck blocks [3,4,6,3]
    - ResNet-101: Bottleneck blocks [3,4,23,3]
    - ResNet-152: Bottleneck blocks [3,8,36,3]

    Training Recommendations:
    ----------------------
    - Optimizer: SGD with momentum (0.9)
    - Learning Rate: Start at 0.1, divide by 10 when error plateaus
    - Weight Decay: 1e-4
    - Batch Size: 256
    - Data Augmentation: Random crops, horizontal flips, color augmentation

    Args:
        num_classes (int): Number of output classes
        in_channels (int): Number of input channels (default: 3 for RGB)
        img_size (int): Size of input images (assumes square input)
        block (nn.Module): Type of residual block to use (Basic or Bottleneck)
        layers (list): Number of blocks in each stage
        zero_init_residual (bool): Whether to initialize residual branch BN to 0
        
    Input:
        - x: tensor of shape (batch_size, in_channels, img_size, img_size)
        
    Output:
        - tensor of shape (batch_size, num_classes)
    """
    
    def __init__(self, num_classes, in_channels=3, img_size=224, 
                 block=ResidualBlock, layers=[2,2,2,2], zero_init_residual=False):
        super().__init__()
        
        self.in_channels = 64
        
        # Initial convolution and pooling
        self.conv1 = nn.Conv2d(in_channels, self.in_channels, kernel_size=7,
                              stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # Calculate final feature map size
        current_size = img_size
        current_size = conv2d_output_size(current_size, kernel_size=7, stride=2, padding=3)  # conv1
        current_size = max_pool_output_size(current_size, pool_ksize=3, pool_stride=2, pool_padding=1)  # maxpool
        for i in range(4):  # 4 residual stages
            current_size = conv2d_output_size(current_size, kernel_size=3, stride=2 if i > 0 else 1, padding=1)
        
        # Final layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
        # Zero-initialize the last BN in each residual branch
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, ResidualBlock):
                    nn.init.constant_(m.bn2.weight, 0)
    
    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )
            
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
            
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

def resnet18(num_classes, in_channels=3, img_size=224, **kwargs):
    """Constructs a ResNet-18 model."""
    return ResNet(num_classes, in_channels, img_size, ResidualBlock, [2, 2, 2, 2], **kwargs)

def resnet34(num_classes, in_channels=3, img_size=224, **kwargs):
    """Constructs a ResNet-34 model."""
    return ResNet(num_classes, in_channels, img_size, ResidualBlock, [3, 4, 6, 3], **kwargs)

def resnet50(num_classes, in_channels=3, img_size=224, **kwargs):
    """Constructs a ResNet-50 model."""
    return ResNet(num_classes, in_channels, img_size, Bottleneck, [3, 4, 6, 3], **kwargs)

def resnet101(num_classes, in_channels=3, img_size=224, **kwargs):
    """Constructs a ResNet-101 model."""
    return ResNet(num_classes, in_channels, img_size, Bottleneck, [3, 4, 23, 3], **kwargs)

def resnet152(num_classes, in_channels=3, img_size=224, **kwargs):
    """Constructs a ResNet-152 model."""
    return ResNet(num_classes, in_channels, img_size, Bottleneck, [3, 8, 36, 3], **kwargs)








