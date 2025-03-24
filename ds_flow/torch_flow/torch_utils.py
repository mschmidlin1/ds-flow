





def Train_Test_Split(train_dataset: torch.utils.data.Dataset, test_dataset: torch.utils.data.Dataset, test_pct = 0.2):
    """
    Takes identical training and testing datasets and subsets them so each has unique examples. 
    The subsets are created at the ratios by using the 'test_pct'.
    """

    class_counts = dict(Counter(train_dataset.targets))

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
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

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
        image = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)
        if self.transform:
            image = self.transform(image)

        return image, label
    

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


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


def conv2d_output_size(img_size, kernel_size=3 , stride=1, padding=0, dilation=1, ):
    return math.floor((img_size+2*padding-dilation*(kernel_size-1)-1)/stride) + 1

def max_pool_output_size(img_size, pool_ksize=2, pool_stride=2, pool_padding=0, dilation=1):
    return math.floor((img_size+(2*pool_padding)-(dilation*pool_ksize-1)-1)/pool_stride) + 1

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

def inference(image, device, model, class_lookup):
    image = image.to(device)
    image = torch.unsqueeze(image, dim=0)
    with torch.inference_mode():
        preds = model(image)
    probs = torch.softmax(preds, dim=1)[0]
    pred_class_idx = torch.argmax(probs).int().item()
    pred_class_label = class_lookup[pred_class_idx]
    return pred_class_label, probs[pred_class_idx].float().item()

def count_model_parameters(model):
    params_list = []
    for l in list(model.parameters()):
        params_list.append(torch.prod(torch.tensor(l.shape)))
    return torch.sum(torch.tensor(params_list))


def split_dataset(dataset, test_pct=0.2, random_seed=0):
    """
    Splits a pytorch dataset using the `test_pct`.
    """
    length = len(dataset)
    all_indices = [i for i in range(length)]
    np.random.seed(random_seed)
    test_indices = np.random.choice(all_indices, size=int(test_pct*length), replace=False)
    train_indices = list(set(all_indices) - set(test_indices))
    print(f"Number test samples: {len(test_indices)}/{length} \tNumber train samples: {len(train_indices)}/{length}")
    test_subset = torch.utils.data.Subset(dataset, test_indices)
    train_subset = torch.utils.data.Subset(dataset, train_indices)
    return test_subset, train_subset


def split_n_load(dataset, test_pct=0.2, batch_size=32, num_workers=-1, device='cuda', shuffle_train=True, random_seed=0) -> tuple[DeviceDataLoader, DeviceDataLoader]:
    """
    Splits the dataset using the `test_pct`.
    Puts the test and train datasets into data loaders.
    Wraps the dataloaders with DeviceDataLoaders to take care of moving to device. 
    """
    # split dataset randomly
    test_subset, train_subset = split_dataset(dataset, test_pct=0.2, random_seed=0)

    #put dataset into data loaders
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size, shuffle=shuffle_train, num_workers=num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(test_subset, batch_size, num_workers=num_workers, pin_memory=True)

    #wrap dataloaders so data is automatically moved to device
    train_loader = DeviceDataLoader(train_loader, device)
    val_loader = DeviceDataLoader(val_loader, device)
    
    return train_loader, val_loader, test_subset, train_subset


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

def fit(epochs: int, model: nn.Module, train_loader: DeviceDataLoader, val_loader: DeviceDataLoader, optimizer, loss_fn, lr_scheduler=None, agg_func=torch.nanmean, secondary_metric=None, secondary_metric_name: str ='', save_file: str = ''):
    """Trains the model. Returns a dictionary of loss histories.
    
    Assumes that the loaders and model are already on the same device. The loader can be "on a device" by wrapping a regular DataLoader with DeviceDataLoader from nn_utils.DeviceDataLoader.


    Parameters
    ----------

    `epochs` : the number of epochs the model will train for.

    `model` : the pytorch model to train.

    `train_loader` : the DataLoader containing the training data. Should be wrapped with DeviceDataLoader so data is automatically loaded to the device.

    `val_loader` : the DataLoader containing the testing data. Should be wrapped with DeviceDataLoader so data is automatically loaded to the device.

    `optimizer` : the pytorch optimizer to train the model with.

    `loss_fn` : custom or standard pytorch loss function.

    `lr_scheduler` : (optional) a pytorch learning rate scheduler to control the learning rate.

    `agg_func` : defaults to `torch.nanmean`. The aggregation of the metrics over all of the batches for a single epoch.

    `secondary_metric` : (optional) a metric that compares predictions to targets. Could be a loss function or some kind of classification metric such as accuracy.

    `secondary_metric_name` : (optional) a string name to call the secondary metric if one is passed.

    `save_file` : (optional) default is an empty string. If a non-empty string is specified it is used to record the best model each epoch.

    Returns
    -------
    `dict` : dictionary of the training and testing losses. Also the secondary metrics of those are specified.
    """
    history = {
        'train_loss': [],
        'test_loss': [],
    }

    if secondary_metric != None:
        history[f'train_{secondary_metric_name}'] = []
        history[f'test_{secondary_metric_name}'] = []
    

        # Pre-training evalutation
        test_loss = model_evaluate(model, val_loader, loss_fn, evaluation_type='testing data', agg_func=agg_func)
        train_loss = model_evaluate(model, train_loader, loss_fn, evaluation_type='training data', agg_func=agg_func)

        secondary_metric_string = ''
        if secondary_metric != None:
            test_metric = model_evaluate(model, val_loader, secondary_metric, evaluation_type=f'testing data {secondary_metric_name}', agg_func=agg_func)
            train_metric = model_evaluate(model, train_loader, secondary_metric, evaluation_type=f'testing data {secondary_metric_name}', agg_func=agg_func)
            history[f'train_{secondary_metric_name}'].append(train_metric)
            history[f'test_{secondary_metric_name}'].append(test_metric)
            secondary_metric_string = f"\tTrain {secondary_metric_name}: {train_metric}\tTest {secondary_metric_name}: {test_metric}"
        
        lr_string = ''
        if lr_scheduler != None:
            lr_string = f"Learning rate: {lr_scheduler.get_last_lr()[0]:.5f}\t"


        print(f"Epoch: {-1}\t{lr_string}Train Loss: {train_loss:.4f}\tTest Loss: {test_loss:.4f}{secondary_metric_string}")
        
        history['train_loss'].append(train_loss)
        history['test_loss'].append((test_loss.cpu()))




    for epoch in range(epochs):
        # Training Phase
        model.train()

        loop = tqdm(train_loader, leave=True, desc=f"Training Epoch {epoch}")
        training_losses = []
        for batch in loop:
            images, y_true = batch
            y_preds = model(images)
            loss = loss_fn(y_preds, y_true)
            training_losses.append(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            optimizer.zero_grad()

        if lr_scheduler != None:
                lr_scheduler.step()
        

        # Validation phase
        test_loss = model_evaluate(model, val_loader, loss_fn, evaluation_type='testing data', agg_func=agg_func)

        secondary_metric_string = ''
        if secondary_metric != None:
            test_metric = model_evaluate(model, val_loader, secondary_metric, evaluation_type=f'testing data {secondary_metric_name}', agg_func=agg_func)
            train_metric = model_evaluate(model, train_loader, secondary_metric, evaluation_type=f'testing data {secondary_metric_name}', agg_func=agg_func)
            history[f'train_{secondary_metric_name}'].append(train_metric)
            history[f'test_{secondary_metric_name}'].append(test_metric)
            secondary_metric_string = f"\tTrain {secondary_metric_name}: {train_metric}\tTest {secondary_metric_name}: {test_metric}"
        
        lr_string = ''
        if lr_scheduler != None:
            lr_string = f"Learning rate: {lr_scheduler.get_last_lr()[0]:.5f}\t"


        print(f"Epoch: {epoch}\t{lr_string}Train Loss: {agg_func(torch.FloatTensor(training_losses)):.4f}\tTest Loss: {test_loss:.4f}{secondary_metric_string}")
        
        if epoch==0 and (save_file != ''):
             torch.save(model.state_dict(), save_file)
             print(f"Saving model state at '{save_file}'")
        elif (test_loss < min(history['test_loss'])) and (save_file != ''):
             torch.save(model.state_dict(), save_file)
             print(f"Saving model state at '{save_file}'")
        
        history['train_loss'].append(agg_func(torch.FloatTensor(training_losses)))
        history['test_loss'].append((test_loss.cpu()))


    
    
    return history

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

def predict_and_visualize(model, device, dataset, class_lookup, idx=None):
    if idx==None:
        idx = np.random.randint(low=0, high=len(dataset), size=1)[0]
    image, label = dataset[idx]
    plt.imshow(image[0], cmap='gray')
    pred_label, probability = inference(image, device=device, model=model, class_lookup=class_lookup)
    print(f"Index: {idx}", 'Label:', class_lookup[label], "\tPredicted:", pred_label, f"   Probability={probability:.2f}")

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