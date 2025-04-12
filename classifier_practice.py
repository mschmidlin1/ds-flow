import datetime
import os
import sys
from typing import Counter

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from ds_flow.general_utils import get_venv_name
from ds_flow.torch_flow import OpenCvImageFolder, get_default_device, get_available_gpus, get_greyscale_classification_transform, \
    get_greyscale_validation_transform, split_transformed_datasets, DeviceDataLoader, fit, accuracy, plot_history, initialize_dataloaders, \
    get_opencv_greyscale_classification_transform, get_opencv_greyscale_validation_transform, get_rgb_classification_transform, get_rgb_validation_transform, \
    VGG
from ds_flow.torch_flow.models import CIFAR10Net, create_basic_cnn, VGGNet, resnet18
import torch.nn as nn
import torchvision

from ds_flow.torch_flow.transforms import get_rgb_classification_transform_light


# Training hyperparameters
EPOCHS = 250  # Increased epochs for better convergence
IMG_SIZE = 32
BATCH_SIZE = 128  # Optimal batch size for VGG (original paper used 256 but we're using smaller images)
LEARNING_RATE = 0.1  # Initial learning rate as per VGG paper
MOMENTUM = 0.9  # Standard momentum value
WEIGHT_DECAY = 5e-4  # L2 regularization as per VGG paper
DROPOUT_RATE = 0.5  # Standard dropout rate for VGG


if __name__ == "__main__":
    # available_gpus = get_available_gpus()
    # print(available_gpus)


    DEVICE = get_default_device()
    print(f"Device name: '{DEVICE}'")
    print(f"Virtual environment name: '{get_venv_name()}'")
    os.makedirs("output", exist_ok=True)
    dttm_str = datetime.datetime.now().__str__().replace(":",".")
    output_dir = f"output/output_{dttm_str}"
    os.makedirs(output_dir, exist_ok=True)

    # train_dataset = OpenCvImageFolder("data/cifar10_combined_images", get_opencv_greyscale_classification_transform())
    # test_dataset = OpenCvImageFolder("data/cifar10_combined_images", get_opencv_greyscale_validation_transform())
    train_dataset = torchvision.datasets.ImageFolder("data/cifar10_combined_images", transform=get_rgb_classification_transform_light(img_size=(IMG_SIZE, IMG_SIZE)))
    test_dataset = torchvision.datasets.ImageFolder("data/cifar10_combined_images", transform=get_rgb_validation_transform(img_size=(IMG_SIZE, IMG_SIZE)))


    class_counts = dict(Counter(train_dataset.targets))
    total = len(train_dataset)
    CLASS_WEIGHTS = 1 - np.array(list(class_counts.values()))/total #you want the weights to be inversely proportional to the 
    CLASS_WEIGHTS = torch.tensor(CLASS_WEIGHTS).to(DEVICE).type(torch.float32)


    train_subset, test_subset = split_transformed_datasets(train_dataset, test_dataset, random_seed=0)

    # Data loaders with optimal batch size
    train_loader = DataLoader(
        train_subset, 
        BATCH_SIZE, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True, 
        persistent_workers=True
    )
    val_loader = DataLoader(
        test_subset, 
        BATCH_SIZE, 
        num_workers=4, 
        pin_memory=True, 
        persistent_workers=True
    )

    train_loader = DeviceDataLoader(train_loader, DEVICE)
    val_loader = DeviceDataLoader(val_loader, DEVICE)

    # Initialize workers for all dataloaders
    init_time = initialize_dataloaders(
        [train_loader, val_loader],
        loader_names=['Training', 'Validation'],
        verbose=False
    )

    # model = create_basic_cnn(len(train_dataset.label_to_int))
    #model = create_basic_cnn(len(train_dataset.classes), img_size=IMG_SIZE, in_channels=3)
    #model = CIFAR10Net(len(train_dataset.classes), in_channels=3, img_size=IMG_SIZE)
    #model = VGGNet(len(train_dataset.classes), in_channels=3, img_size=IMG_SIZE, config='VGG16', batch_norm=True)
    model = resnet18(len(train_dataset.classes), in_channels=3, img_size=IMG_SIZE)
    #model = VGG('VGG16')
    model.to(DEVICE)

    # Optimizer configurations
    # Original VGG paper optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY
    )
    
    # Alternative optimizers (commented out)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.1)

    # Loss function with class weights
    loss_fn = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS)

    # Learning rate schedulers
    # Option 1: OneCycleLR (modern approach, often works well)
    # lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #     optimizer,
    #     max_lr=LEARNING_RATE,
    #     steps_per_epoch=len(train_loader),
    #     epochs=EPOCHS,
    #     anneal_strategy='cos',
    #     pct_start=0.3  # Spend 30% of training increasing LR
    # )
    
    # Option 2: ReduceLROnPlateau (original VGG approach)
    # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer,
    #     mode='min',
    #     factor=0.1,
    #     patience=5,
    #     verbose=True
    # )
    
    # Option 3: CosineAnnealingLR
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS
    )

    dttm_str = datetime.datetime.now().__str__().replace(":",".")

    history = fit(
        EPOCHS, 
        model, 
        train_loader, 
        val_loader, 
        optimizer, 
        loss_fn, 
        lr_scheduler=lr_scheduler,  
        secondary_metric=accuracy, 
        secondary_metric_name="accuracy",
        save_file=f"{output_dir}/weights.pth"
    )

    fig, ax = plot_history(history)
    fig.savefig(f"{output_dir}/results.png")

    hist_df = pd.DataFrame(history)
    hist_df = hist_df.map(float)
    hist_df.to_pickle(f"{output_dir}/results.pkl")



    # #save weights as C# readable
    # import io
    # import torch
    # import leb128

    # def _elem_type(t):
    #     dt = t.dtype

    #     if dt == torch.uint8:
    #         return 0
    #     elif dt == torch.int8:
    #         return 1
    #     elif dt == torch.int16:
    #         return 2
    #     elif dt == torch.int32:
    #         return 3
    #     elif dt == torch.int64:
    #         return 4
    #     elif dt == torch.float16:
    #         return 5
    #     elif dt == torch.float32:
    #         return 6
    #     elif dt == torch.float64:
    #         return 7
    #     elif dt == torch.bool:
    #         return 11
    #     elif dt == torch.bfloat16:
    #         return 15
    #     else:
    #         return 4711

    # def _write_tensor(t, stream):
    #     stream.write(leb128.u.encode(_elem_type(t)))
    #     stream.write(leb128.u.encode(len(t.shape)))
    #     for s in t.shape:
    #         stream.write(leb128.u.encode(s))
    #     stream.write(t.numpy().tobytes())

    # def save_state_dict(sd, stream):
    #     """
    #     Saves a PyToch state dictionary using the format that TorchSharp can
    #     read.
    #     :param sd: A dictionary produced by 'model.state_dict()'
    #     :param stream: An write stream opened for binary I/O.
    #     """
    #     stream.write(leb128.u.encode(len(sd)))
    #     for entry in sd:
    #         stream.write(leb128.u.encode(len(entry)))
    #         stream.write(bytes(entry, 'utf-8'))
    #         _write_tensor(sd[entry], stream)
    # cs_weights_file = f"weights.dat"

    # f = open(cs_weights_file, "wb")
    # save_state_dict(model.to("cpu").state_dict(), f)
    # f.close()