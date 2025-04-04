import torch
import torchvision
import torchvision.transforms as transforms
import os
from PIL import Image

def cifar10_to_combined_image_folders(data_root='./data/cifar10_combined_images'):
    """Combines train and test CIFAR-10 data into single class folders.

    Args:
        data_root (str, optional): Root directory to store the combined image folders. Defaults to './data/cifar10_combined_images'.
    """

    transform = transforms.ToTensor()

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    classes = trainset.classes

    # Create class directories
    for class_name in classes:
        os.makedirs(os.path.join(data_root, class_name), exist_ok=True)

    # Save training images
    train_count = 0
    for image, label in trainset:
        class_name = classes[label]
        image_pil = transforms.ToPILImage()(image)
        image_pil.save(os.path.join(data_root, class_name, f'train_{train_count}.png'))
        train_count += 1

    # Save test images
    test_count = 0
    for image, label in testset:
        class_name = classes[label]
        image_pil = transforms.ToPILImage()(image)
        image_pil.save(os.path.join(data_root, class_name, f'test_{test_count}.png'))
        test_count += 1

def cifar100_to_combined_image_folders(data_root='./data/cifar100_combined_images'):
    """Combines train and test CIFAR-100 data into single class folders.

    Args:
        data_root (str, optional): Root directory to store the combined image folders. Defaults to './data/cifar100_combined_images'.
    """
    import pickle

    transform = transforms.ToTensor()

    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

    # Load class names
    with open(os.path.join('./data', 'cifar-100-python', 'meta'), 'rb') as f:
        data = pickle.load(f, encoding='bytes')
        classes = [x.decode('utf-8') for x in data[b'fine_label_names']]

    # Create class directories
    for class_name in classes:
        os.makedirs(os.path.join(data_root, class_name), exist_ok=True)

    # Save training images
    train_count = 0
    for image, label in trainset:
        class_name = classes[label]
        image_pil = transforms.ToPILImage()(image)
        image_pil.save(os.path.join(data_root, class_name, f'train_{train_count}.png'))
        train_count += 1

    # Save test images
    test_count = 0
    for image, label in testset:
        class_name = classes[label]
        image_pil = transforms.ToPILImage()(image)
        image_pil.save(os.path.join(data_root, class_name, f'test_{test_count}.png'))
        test_count += 1

# Example Usage:
cifar10_to_combined_image_folders()
cifar100_to_combined_image_folders()

# Then, you can use ImageFolder:
# dataset = torchvision.datasets.ImageFolder(root='./data/cifar10_combined_images', transform=transforms.ToTensor())