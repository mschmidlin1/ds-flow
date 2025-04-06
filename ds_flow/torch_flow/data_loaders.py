import time
from ds_flow.torch_flow.torch_utils import to_device


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


def initialize_dataloaders(loaders, loader_names=None, verbose=True):
    """Initialize workers for multiple DataLoaders by triggering a dummy iteration through each.
    
    This function is particularly useful when using DataLoaders with persistent_workers=True.
    It explicitly triggers the worker initialization process for each DataLoader before any
    actual data loading begins. This moves the initialization overhead to a controlled point
    in your code rather than having it occur during the first actual use of the DataLoader.
    
    Args:
        loaders (list): List of DataLoader or DeviceDataLoader objects to initialize
        loader_names (list, optional): List of names for each loader for verbose output.
            If None, loaders will be referred to by index. Defaults to None.
        verbose (bool, optional): Whether to print timing information. Defaults to True.
    
    Returns:
        float: Total time taken for initialization in seconds
        
    Example:
        >>> train_loader = DataLoader(train_data, num_workers=4, persistent_workers=True)
        >>> val_loader = DataLoader(val_data, num_workers=4, persistent_workers=True)
        >>> init_time = initialize_dataloaders(
        ...     [train_loader, val_loader],
        ...     loader_names=['Training', 'Validation']
        ... )
        Initializing DataLoader workers...
        Initializing Training loader workers...
        Initializing Validation loader workers...
        Worker initialization took 15.23 seconds
    
    Notes:
        - This is most beneficial when using DataLoaders with persistent_workers=True
        - The initialization process includes:
            * Spawning worker processes
            * Loading required libraries in worker processes
            * Setting up worker state
            * Loading initial data
        - On Windows, initialization can be particularly slow due to process spawning overhead
        - Each worker process needs to import all required libraries (e.g., OpenCV, PyTorch)
    """
    if loader_names is None:
        loader_names = [f"DataLoader_{i}" for i in range(len(loaders))]
    
    if verbose:
        print("\nInitializing DataLoader workers...")
    
    init_start = time.time()
    
    for loader, name in zip(loaders, loader_names):
        if verbose:
            print(f"Initializing {name} loader workers...")
        
        # Trigger worker initialization by attempting to get first batch
        iterator = iter(loader)
        try:
            next(iterator)
        except StopIteration:
            if verbose:
                print(f"Warning: {name} loader is empty")
    
    total_time = time.time() - init_start
    if verbose:
        print(f"Worker initialization took {total_time:.2f} seconds")
    
    return total_time