import numpy as np
import torch
import torchvision
import cv2

class OpenCvGrayscale:
    """
    Transform that converts OpenCV images to grayscale.
    Works with numpy arrays (OpenCV images) instead of PIL images.
    """
    def __call__(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 2:
            return image
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

class OpenCvToTensor:
    """
    Transform that converts OpenCV images (numpy arrays) to PyTorch tensors.
    This is similar to torchvision.transforms.ToTensor() but works with OpenCV images.
    The main differences are:
    1. Works with numpy arrays instead of PIL images
    2. Handles both RGB and grayscale images
    3. Preserves the original data type (uint8, uint16, etc.)
    """
    def __call__(self, image: np.ndarray) -> torch.Tensor:
        # Handle grayscale images
        if len(image.shape) == 2:
            image = image[..., None]
        
        # Convert from HWC to CHW format
        image = image.transpose(2, 0, 1)
        
        # Convert to tensor and normalize to [0, 1]
        image = torch.from_numpy(image).float()
        if image.dtype == torch.uint8:
            image = image / 255.0
        elif image.dtype == torch.uint16:
            image = image / 65535.0
            
        return image


def get_opencv_greyscale_classification_transform(img_size=(32, 32)):
    """
    Creates a transform pipeline for grayscale image classification using OpenCV.
    This transform:
    1. Converts OpenCV BGR images to grayscale
    2. Converts to PyTorch tensor
    3. Applies standard data augmentation techniques
    
    Args:
        img_size (tuple): Target size for the output images (height, width). Defaults to (32, 32).
        
    Returns:
        torchvision.transforms.Compose: A composed transform pipeline for grayscale image processing
    """
    return torchvision.transforms.Compose([
                        OpenCvGrayscale(),  # First convert to grayscale using OpenCV
                        OpenCvToTensor(),   # Then convert to tensor
                        torchvision.transforms.Resize(img_size, antialias=True),
                        torchvision.transforms.RandomCrop(img_size, padding=4, padding_mode='reflect'), 
                        torchvision.transforms.RandomHorizontalFlip(),
                        torchvision.transforms.RandomVerticalFlip(), 
                        torchvision.transforms.RandomRotation(degrees=(0, 180))
                        ])

def get_opencv_rgb_classification_transform(img_size=(32, 32)):
    """
    Creates a transform pipeline for RGB image classification using OpenCV.
    This transform:
    1. Converts OpenCV BGR images to RGB format
    2. Converts to PyTorch tensor
    3. Applies standard data augmentation techniques
    """
    return torchvision.transforms.Compose([
                        OpenCvToTensor(),   # Convert to tensor (already handles RGB)
                        torchvision.transforms.Resize(img_size, antialias=True),
                        torchvision.transforms.RandomCrop(img_size, padding=4, padding_mode='reflect'),
                        torchvision.transforms.RandomHorizontalFlip(),
                        torchvision.transforms.RandomVerticalFlip(),
                        torchvision.transforms.RandomRotation(degrees=(0, 180)),
                        torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
                        ])

def get_opencv_greyscale_validation_transform(img_size=(32, 32)):
    """
    Creates a transform pipeline for grayscale image validation using OpenCV.
    This transform only includes essential preprocessing without augmentation:
    1. Converts OpenCV BGR images to grayscale
    2. Converts to PyTorch tensor
    3. Resizes to target size
    
    Args:
        img_size (tuple): Target size for the output images (height, width). Defaults to (32, 32).
        
    Returns:
        torchvision.transforms.Compose: A composed transform pipeline for grayscale image processing
    """
    return torchvision.transforms.Compose([
                        OpenCvGrayscale(),  # First convert to grayscale using OpenCV
                        OpenCvToTensor(),   # Then convert to tensor
                        torchvision.transforms.Resize(img_size, antialias=True)
                        ])

def get_opencv_rgb_validation_transform(img_size=(32, 32)):
    """
    Creates a transform pipeline for RGB image validation using OpenCV.
    This transform only includes essential preprocessing without augmentation:
    1. Converts OpenCV BGR images to RGB format
    2. Converts to PyTorch tensor
    3. Resizes to target size
    
    Args:
        img_size (tuple): Target size for the output images (height, width). Defaults to (32, 32).
        
    Returns:
        torchvision.transforms.Compose: A composed transform pipeline for RGB image processing
    """
    return torchvision.transforms.Compose([
                        OpenCvToTensor(),   # Convert to tensor (already handles RGB)
                        torchvision.transforms.Resize(img_size, antialias=True)
                        ])

def get_greyscale_classification_transform(img_size=(32, 32)):
    """
    Creates a transform pipeline for grayscale image classification using PIL/PyTorch.
    This transform:
    1. Converts PIL images to grayscale
    2. Converts to PyTorch tensor
    3. Applies standard data augmentation techniques
    
    Args:
        img_size (tuple): Target size for the output images (height, width). Defaults to (32, 32).
        
    Returns:
        torchvision.transforms.Compose: A composed transform pipeline for grayscale image processing
    """
    return torchvision.transforms.Compose([
                        torchvision.transforms.Grayscale(),
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Resize(img_size, antialias=True),
                        torchvision.transforms.RandomCrop(img_size, padding=4, padding_mode='reflect'), 
                        torchvision.transforms.RandomHorizontalFlip(),
                        torchvision.transforms.RandomVerticalFlip(), 
                        torchvision.transforms.RandomRotation(degrees=(0, 180))
                        ])

def get_rgb_classification_transform(img_size=(32, 32)):
    """
    Creates a transform pipeline for RGB image classification using PIL/PyTorch.
    This transform:
    1. Converts to PyTorch tensor
    2. Applies standard data augmentation techniques
    
    Args:
        img_size (tuple): Target size for the output images (height, width). Defaults to (32, 32).
        
    Returns:
        torchvision.transforms.Compose: A composed transform pipeline for RGB image processing
    """
    return torchvision.transforms.Compose([
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Resize(img_size, antialias=True),
                        torchvision.transforms.RandomCrop(img_size, padding=4, padding_mode='reflect'),
                        torchvision.transforms.RandomHorizontalFlip(),
                        torchvision.transforms.RandomVerticalFlip(),
                        torchvision.transforms.RandomRotation(degrees=(0, 180)),
                        torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
                        ])
def get_rgb_classification_transform_light(img_size=(32, 32)):
    """
    Creates a transform pipeline for RGB image classification using PIL/PyTorch.
    This transform:
    1. Converts to PyTorch tensor
    2. Applies standard data augmentation techniques
    
    Args:
        img_size (tuple): Target size for the output images (height, width). Defaults to (32, 32).
        
    Returns:
        torchvision.transforms.Compose: A composed transform pipeline for RGB image processing
    """
    return torchvision.transforms.Compose([
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Resize(img_size, antialias=True),
                        torchvision.transforms.RandomHorizontalFlip(),
                        torchvision.transforms.RandomVerticalFlip(),
                        torchvision.transforms.RandomRotation(degrees=(0, 180))
                        ])

def get_greyscale_validation_transform(img_size=(32, 32)):
    """
    Creates a transform pipeline for grayscale image validation using PIL/PyTorch.
    This transform only includes essential preprocessing without augmentation:
    1. Converts PIL images to grayscale
    2. Converts to PyTorch tensor
    3. Resizes to target size
    
    Args:
        img_size (tuple): Target size for the output images (height, width). Defaults to (32, 32).
        
    Returns:
        torchvision.transforms.Compose: A composed transform pipeline for grayscale image processing
    """
    return torchvision.transforms.Compose([
                        torchvision.transforms.Grayscale(),
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Resize(img_size, antialias=True)
                        ])

def get_rgb_validation_transform(img_size=(32, 32)):
    """
    Creates a transform pipeline for RGB image validation using PIL/PyTorch.
    This transform only includes essential preprocessing without augmentation:
    1. Converts to PyTorch tensor
    2. Resizes to target size
    
    Args:
        img_size (tuple): Target size for the output images (height, width). Defaults to (32, 32).
        
    Returns:
        torchvision.transforms.Compose: A composed transform pipeline for RGB image processing
    """
    return torchvision.transforms.Compose([
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Resize(img_size, antialias=True)
                        ])