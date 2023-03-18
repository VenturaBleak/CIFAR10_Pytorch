#%%
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchinfo import summary

from data_setup import get_dataloaders_cifar10
from models import model_choice
from metrics import get_model_path
#%%
# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
print('Device:', device)
#%%
# Data parameters - keep as is
random_seed = 42
num_classes_user = 10
batch_size = 32
augment = True
validation_fraction = 0.1
#%%
# Model hyperparameters
num_epochs = 100
model_name = 'NiN'
pretrained = False
optimizer_choice = 'SGD'
scheduler_choice = 'cosine'
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
# Instantiate optimizer
if optimizer_choice == 'SGD':
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
elif optimizer_choice == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
else:
    raise ValueError('Optimizer_choice not supported')
#%%
# Instantiate scheduler
if scheduler_choice == 'reduce_on_plateau':
    # scheduler hyperparams
    mode = 'min'
    factor = 0.2
    patience = 10
    cooldown = 5
    min_lr = 1e-6
    verbose = True
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                            mode=mode,
                                                            factor=factor,
                                                            patience=patience,
                                                            cooldown=cooldown,
                                                            min_lr=min_lr,
                                                            verbose=verbose)
elif scheduler_choice == 'cosine':
    # scheduler hyperparams
    T_max = num_epochs # he number of epochs or iterations to complete one cosine annealing cycle.
    eta_min = 1e-6 # The minimum learning rate at the end of each cycle
    # scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                           T_max=T_max,
                                                           eta_min=eta_min)
elif scheduler_choice == 'cosine_warm_restarts':
    # scheduler hyperparams
    T_0 = 10 # the number of epochs or iterations for the initial restart cycle
    T_mult = 2 # The factor by which the cycle length increases after each restart.
    eta_min = 1e-6 # min learning rat at the end of each cycle
    # scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                     T_0=T_0,
                                                                     T_mult=T_mult,
                                                                     eta_min=eta_min)

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
assert images.shape[0] == batch_size
assert labels.shape[0] == batch_size
assert images.shape[1] == 3 # RGB
assert images.shape[2] == resolution
assert images.shape[3] == resolution
#%%
# For the given batch, check that the channel means and standard deviations are roughly 0 and 1, respectively:
print('Channel mean:', torch.mean(images[:, 0, :, :]))
print('Channel std:', torch.std(images[:, 0, :, :]))
#%%
# visualize some sample images
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
model_folder_name = 'models'
# get current working directory
current_working_directory = os.getcwd()
model_dir = os.path.join(current_working_directory, model_folder_name)

# get model path
model_path = get_model_path(model_dir, model_name)

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
                                        best_model_save_path=model_path,
                                        num_classes=num_classes)
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

