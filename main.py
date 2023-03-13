#%%
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchinfo import summary
from torchvision import transforms

from data_setup import get_readers1, get_readers2, train_mean_std, get_loaders
from models import model_choice
#%%
# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
print('Device:', device)
#%% md
## 1. Load Data
#%%
# Hyperparameters
random_seed = 42
learning_rate = 0.001
num_epochs = 9
batch_size = 32
model_name = 'NiN'
pretrained = True
optimizer_choice = 'Adam'
scheduler_choice = 'cyclic'
augment = True
lr = 0.01
base_lr= 0.005
max_lr = 0.01
step_size_up = 3
mode="exp_range"
gamma=0.85
#%%
# to reduce training time, set train_indices to a subset of the training data
# https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/cnn/nin-cifar10_batchnorm.ipynb
if str(device) != 'cpu':
    # all training data
    train_indices = None
else:
    # subset of training data
    train_indices = torch.arange(0, 100)
#%%
# get data sets and classes_to_idx
train_reader, test_reader, classes_to_idx = get_readers1(train_indices=train_indices)
# get mean and std
train_mean, train_std = train_mean_std(train_reader, batch_size=batch_size)
#%%
# get num classes
num_classes = len(classes_to_idx.keys())
# display classes
if num_classes <= 10:
    print(classes_to_idx)
else:
    print('Too many classes to print. Number of classes:', num_classes)
#%%
# get required resolution for chosen model
torch.manual_seed(random_seed)
model, resolution = model_choice(model_name, pretrained, num_classes)
model = model.to(device)
#%%
# Create optimizer and scheduler
if optimizer_choice == 'SGD':
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
elif optimizer_choice == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    cycle_momentum = False
else:
    raise ValueError('Optimizer_choice not supported')

#%%
### Scheduler
# https://www.kaggle.com/code/isbhargav/guide-to-pytorch-learning-rate-scheduling
if scheduler_choice == 'cyclic':
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,
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
# inspect model
summary(model, input_size=[1, 3, resolution, resolution])
#%%
# Create training transform with TrivialAugment
# Create testing transform (no data augmentation)
test_transform = transforms.Compose([
    transforms.Resize((resolution, resolution)),
    transforms.ToTensor(),
    transforms.Normalize(mean=train_mean, std=train_std)
])

if augment == True:
    train_transform = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.TrivialAugmentWide(num_magnitude_bins=6),
        transforms.ToTensor(),
        transforms.Normalize(mean=train_mean, std=train_std)
    ])
    print('TrivialAugmentWide applied')
else:
    train_transform = test_transform

#%%
# Load data
train_reader_aug, train_reader, test_reader = get_readers2(train_transform=train_transform,
                                                           test_transform=test_transform,
                                                           train_indices=train_indices)

train_loader_aug, train_loader, test_loader = get_loaders(batch_size=batch_size,
                                                          device=device,
                                                          train_reader_aug = train_reader_aug,
                                                          train_reader=train_reader,
                                                          test_reader=test_reader)
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
from engine import train_classifier_simple_v1
log_dict = train_classifier_simple_v1(num_epochs=num_epochs,
                                      model=model,
                                      scheduler=scheduler,
                                      train_loader_aug=train_loader_aug,
                                      train_loader=train_loader,
                                      test_loader=test_loader,
                                      optimizer=optimizer,
                                      device=device)

#%%
### Evaluate the model


# plot values of learning rate key in log_dict
plt.plot(range(0, len(log_dict['learning_rate_per_epoch'])),log_dict['learning_rate_per_epoch'])
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.show()