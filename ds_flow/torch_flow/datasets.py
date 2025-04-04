import os
from typing import Counter
import torch
import numpy as np
import cv2

from ds_flow.torch_flow.data_loaders import DeviceDataLoader

def split_n_load(dataset, test_pct=0.2, batch_size=32, num_workers=-1, device='cuda', shuffle_train=True, random_seed=0) -> tuple[DeviceDataLoader, DeviceDataLoader]:
    """
    Splits the dataset using the `test_pct`.
    Puts the test and train datasets into data loaders.
    Wraps the dataloaders with DeviceDataLoaders to take care of moving to device. 
    """
    # split dataset randomly
    test_subset, train_subset = split_single_dataset(dataset, test_pct=0.2, random_seed=0)

    #put dataset into data loaders
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size, shuffle=shuffle_train, num_workers=num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(test_subset, batch_size, num_workers=num_workers, pin_memory=True)

    #wrap dataloaders so data is automatically moved to device
    train_loader = DeviceDataLoader(train_loader, device)
    val_loader = DeviceDataLoader(val_loader, device)
    
    return train_loader, val_loader, test_subset, train_subset


def split_single_dataset(dataset, test_pct=0.2, random_seed=0):
    """
    Splits a single PyTorch dataset into train and test subsets using random sampling.
    
    Args:
        dataset: PyTorch Dataset to split
        test_pct (float): Percentage of data to use for testing (between 0 and 1)
        random_seed (int): Random seed for reproducible splits
    
    Returns:
        tuple: (test_subset, train_subset) - PyTorch Subset objects
    """
    # ... existing code ...
    length = len(dataset)
    all_indices = [i for i in range(length)]
    np.random.seed(random_seed)
    test_indices = np.random.choice(all_indices, size=int(test_pct*length), replace=False)
    train_indices = list(set(all_indices) - set(test_indices))
    test_subset = torch.utils.data.Subset(dataset, test_indices)
    train_subset = torch.utils.data.Subset(dataset, train_indices)
    return test_subset, train_subset


def split_transformed_datasets(train_dataset: torch.utils.data.Dataset, 
                             test_dataset: torch.utils.data.Dataset, 
                             test_pct=0.2,
                             random_seed=None):
    """
    Splits two identical datasets that have different transforms applied. This is useful when
    you want different data augmentation/preprocessing for training vs testing, but need to ensure
    the split uses the same indices for both datasets.
    
    Args:
        train_dataset: Dataset with training transforms applied
        test_dataset: Same dataset but with testing transforms applied
        test_pct (float): Percentage of data to use for testing (between 0 and 1)
        random_seed (Optional[int]): Random seed for reproducible splits
    
    Returns:
        tuple: (train_subset, test_subset) - PyTorch Subset objects with respective transforms
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    length = len(train_dataset)
    all_indices = [i for i in range(length)]
    test_indices = np.random.choice(all_indices, size=int(test_pct*length), replace=False)
    train_indices = list(set(all_indices) - set(test_indices))

    test_subset = torch.utils.data.Subset(test_dataset, test_indices)
    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    return train_subset, test_subset


class OpenCvImageFolder(torch.utils.data.Dataset):
    """
    This is an implementation of a pytorch Dataset very similar to the pytorch "Image Folder". 
    The main difference is that it uses open-cv to read in images rather than PIL. 
    """
    def __init__(self, root_dir, transform=None, read_type=cv2.IMREAD_ANYDEPTH):
        self.root_dir = root_dir
        self.transform = transform
        self.read_type = read_type

        # Get list of all image files in root_dir
        self.image_paths = []
        for root, _, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith(".jpg") or file.endswith(".png"):
                    self.image_paths.append(os.path.join(root, file))

        # Label each image with the name of its parent directory
        self.labels = [os.path.basename(os.path.dirname(path)) for path in self.image_paths]

        unique_labels = set(self.labels)
        self.label_to_int = {label: i for i, label in enumerate(unique_labels)}
        self.targets = [self.label_to_int[label] for label in self.labels]
        self.labels = self.targets

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Open image and apply transforms if any
        image = cv2.imread(img_path, self.read_type)
        if self.transform:
            image = self.transform(image)

        return image, label


