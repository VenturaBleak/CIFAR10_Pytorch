#%%
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchinfo import summary

from data_setup import get_dataloaders_cifar10
from models import model_choice
#%%
# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
print('Device:', device)
#%% md
## 1. Load Data
#%%
# Data parameters
random_seed = 42
num_classes_user = 10
batch_size = 32
augment = True
validation_fraction = 0.1
# Model hyperparameters
num_epochs = 5
model_name = 'NiN'
pretrained = True
optimizer_choice = 'SGD'
scheduler_choice = 'cyclic'
lr = 0.01
#%%
# get required resolution for chosen model
torch.manual_seed(random_seed)
model, resolution = model_choice(model_name, pretrained, num_classes_user)
model = model.to(device)
#%%
# inspect model
summary(model, input_size=[1, 3, resolution, resolution])
#%%
# Create optimizer and scheduler
if optimizer_choice == 'SGD':
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    cycle_momentum = True
elif optimizer_choice == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    cycle_momentum = False
else:
    raise ValueError('Optimizer_choice not supported')
#%%
### Scheduler
# https://www.kaggle.com/code/isbhargav/guide-to-pytorch-learning-rate-scheduling
if scheduler_choice == 'cyclic':
    # scheduler hyperparams
    base_lr = 0.00005
    max_lr = 0.001
    step_size_up = 5
    mode = 'triangular2'
    gamma = 0.90
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer=optimizer,
                                                  base_lr=base_lr,
                                                  max_lr=max_lr,
                                                  step_size_up=step_size_up,
                                                  mode=mode,
                                                  gamma=gamma,
                                                  cycle_momentum=cycle_momentum)
elif scheduler_choice == None:
    scheduler = None
else:
    raise ValueError('Scheduler_choice not supported')
#%%
# Load data
train_loader_aug, train_loader, \
    valid_loader, test_loader, \
    classes_to_idx = get_dataloaders_cifar10(batch_size=batch_size,
                                             device=device,
                                             resolution=resolution,
                                             augmentation=True,
                                             validation_fraction=validation_fraction)
#%%
# get num classes
num_classes = len(classes_to_idx.keys())
assert num_classes == num_classes_user
# display classes
if num_classes <= 10:
    print(classes_to_idx)
else:
    print('Too many classes to print. Number of classes:', num_classes)
#%%
# Checking the dataset
images, labels = next(iter(train_loader_aug))
print('Image batch dimensions:', images.shape)
print('Image label dimensions:', labels.shape)
#%%
# For the given batch, check that the channel means and standard deviations are roughly 0 and 1, respectively:
print('Channel mean:', torch.mean(images[:, 0, :, :]))
print('Channel std:', torch.std(images[:, 0, :, :]))
#%%
from torchvision.utils import make_grid
plt.figure(figsize=(8, 8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(make_grid(
    images[:batch_size],
    padding=2, normalize=True),
    (1, 2, 0)))
plt.show();
#%% md
## 2. Train
#%%
# mkdir for saving model, if it doesn't exist
import os
mo = 'models'
# get current working directory
cwd = os.getcwd()
model_dir = os.path.join(cwd, mo)
model_path = os.path.join(model_dir, model_name + ".pt")

# Check if the directory exists
if not os.path.exists(model_dir):
  # Create the directory
  os.makedirs(model_dir)
#%%
from engine import train_classifier_simple_v1, train_classifier_simple_v2
log_dict = train_classifier_simple_v2(num_epochs=num_epochs,
                                        model=model,
                                        scheduler=scheduler,
                                        train_loader_aug=train_loader_aug,
                                        train_loader=train_loader,
                                        valid_loader=valid_loader,
                                        test_loader=None,
                                        optimizer=optimizer,
                                        device=device,
                                        best_model_save_path=model_path)
#%%
# log_dict = train_classifier_simple_v1(num_epochs=num_epochs,
#                                       model=model,
#                                       scheduler=scheduler,
#                                       train_loader_aug=train_loader_aug,
#                                       train_loader=train_loader,
#                                       test_loader=test_loader,
#                                       optimizer=optimizer,
#                                       device=device)
#%%
# save log_dict as pickle
import pickle
log_dict_path = os.path.join(model_dir, model_name + "_log_dict.pkl")
# save log_dict
with open(log_dict_path, "wb") as f:
    pickle.dump(log_dict, f)