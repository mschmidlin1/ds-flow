

# import os
# import sys

# from ds_flow.torch_flow.torch_utils import OpenCvImageFolder


# def get_venv_name():
#     # Check if the script is running inside a virtual environment
#     if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
#         # Get the path to the virtual environment and extract its name
#         venv_path = sys.prefix
#         venv_name = os.path.basename(venv_path)
#         return venv_name
#     else:
#         return None





# available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
# available_gpus


# DEVICE = my_nn_utils.get_default_device()
# print(f"Device name: '{DEVICE}'")
# print(f"Virtual environment name: '{environment_utils.get_venv_name()}'")

# train_dataset = OpenCvImageFolder("", Transform())
# test_dataset = OpenCvImageFolder("", Transform())

# class_counts = dict(Counter(train_dataset.targets))
# total = len(train_dataset)
# CLASS_WEIGHTS = 1 - np.array(list(class_counts.values()))/total #you want the weights to be inversely proportional to the 
# CLASS_WEIGHTS = torch.tensor(CLASS_WEIGHTS).to(DEVICE).type(torch.float32)


# train_subset, test_subset = my_datasets.Train_Test_Split(train_dataset, test_dataset, VAL_PCT)

# train_loader = DataLoader(train_subset, BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
# val_loader = DataLoader(test_subset, BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)

# train_loader = my_nn_utils.DeviceDataLoader(train_loader, DEVICE)
# val_loader = my_nn_utils.DeviceDataLoader(val_loader, DEVICE)






# model = models.CNN2(img_size=IMG_SIZE, num_classes=NUM_CLASSES)
# model.to(DEVICE)
# #optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.1)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# CLASS_WEIGHTS = torch.tensor(CLASS_WEIGHTS).to(DEVICE).type(torch.float32)

# loss_fn = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS)

# #lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(train_loader), epochs=EPOCHS, anneal_strategy='cos')

# dttm_str = datetime.datetime.now().__str__().replace(":",".")

# history = training.fit(
#     EPOCHS, 
#     model, 
#     train_loader, 
#     val_loader, 
#     optimizer, 
#     loss_fn, 
#     None, 
#     secondary_metric=evaluation.accuracy, 
#     secondary_metric_name="accuracy",
#     save_file="file.pth")

# fig, ax = visualizations.plot_history(history)
# fig.savefig("results.png")

# hist_df = pd.DataFrame(history)
# hist_df = hist_df.applymap(float)
# hist_df.to_pickle("results.pkl")



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
