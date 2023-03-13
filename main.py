#%%
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchinfo import summary
from torchvision import transforms

from data_setup import train_mean_std

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device:', device)
#%% md
## 1. Load Data
#%%
# Hyperparameters
random_seed = 42
learning_rate = 0.001
num_epochs = 5
batch_size = 32
model = 'NiN'
pretrained = True
GRAYSCALE = False
#%%
# to reduce training time, set train_indices to a subset of the training data
# https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/cnn/nin-cifar10_batchnorm.ipynb
# train_indices = None
train_indices = torch.arange(0, 1000)
#%%
# get standard mean and std for training data

train_mean, train_std, classes_to_idx = train_mean_std(batch_size=batch_size,
                                                       train_indices=train_indices)

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
from models import model_choice
torch.manual_seed(random_seed)
model, resolution = model_choice(model, pretrained, num_classes)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#%%
# inspect model
summary(model, input_size=[1, 3, resolution, resolution])
#%%
# Create training transform with TrivialAugment
train_transform_trivial_augment = transforms.Compose([
    transforms.Resize((resolution, resolution)),
    transforms.TrivialAugmentWide(num_magnitude_bins=6),
    transforms.ToTensor(),
    transforms.Normalize(mean=train_mean, std=train_std)
])
# Create testing transform (no data augmentation)
test_transform = transforms.Compose([
    transforms.Resize((resolution, resolution)),
    transforms.ToTensor(),
    transforms.Normalize(mean=train_mean, std=train_std)
])
#%%
# Load data
from data_setup import load_data
train_loader, test_loader = load_data(batch_size=batch_size,
                                      train_transform = train_transform_trivial_augment,
                                      test_transform = test_transform,
                                      train_indices=train_indices)
#%%
# Checking the dataset
images, labels = next(iter(train_loader))
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
from eval import compute_accuracy
import time
start_time = time.time()
for epoch in range(num_epochs):

    model.train()

    for batch_idx, (features, targets) in enumerate(train_loader):

        ### PREPARE MINIBATCH
        features = features.to(device)
        targets = targets.to(device)

        ### FORWARD AND BACK PROP
        logits, probas = model(features)
        loss = F.cross_entropy(logits, targets)
        optimizer.zero_grad()

        loss.backward()

        ### UPDATE MODEL PARAMETERS
        optimizer.step()

        ### LOGGING
        if batch_idx % int(len(train_loader)*(1/5)) == 0:
            print(f'Epoch: {epoch + 1:03d}/{num_epochs:03d} | '
                  f'Batch {batch_idx:03d}/{len(train_loader):03d} |'
                  f' Loss: {loss:.4f}')

    # no need to build the computation graph for backprop when computing accuracy
    with torch.set_grad_enabled(False):
        train_acc = compute_accuracy(model, train_loader, device=device)
        print(f'Epoch: {epoch + 1:03d}/{num_epochs:03d} Train Acc.: {train_acc:.2f}%')

    elapsed = (time.time() - start_time) / 60
    print(f'Time elapsed: {elapsed:.2f} min')

elapsed = (time.time() - start_time) / 60
print(f'Total Training Time: {elapsed:.2f} min')
#%%
# test
with torch.set_grad_enabled(False):
    test_acc = compute_accuracy(model, test_loader, device=device)
    print(f'Test Accuracy: {test_acc:.2f}%')