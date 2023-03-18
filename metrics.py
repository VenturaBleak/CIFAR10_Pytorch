import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchmetrics
import os
import numpy as np

def compute_accuracy(model, data_loader, device):
    # ToDo: can be deleted
    model.eval()
    correct_pred, num_examples = 0, 0
    with torch.inference_mode():
        for i, (features, targets) in enumerate(data_loader):
            features = features.to(device)
            targets = targets.to(device)

            logits, probas = model(features)
            _, predicted_labels = torch.max(probas, 1)
            num_examples += targets.size(0)
            correct_pred += (predicted_labels == targets).sum()
        acc = correct_pred.float() / num_examples * 100

        return acc

def compute_epoch_loss(model, data_loader, device):
    # ToDo: can be deleted
    model.eval()
    curr_loss, num_examples = 0., 0
    with torch.inference_mode():
        for features, targets in data_loader:
            features = features.to(device)
            targets = targets.to(device)
            logits, probas = model(features)
            # if isinstance(logits, torch.distributed.rpc.api.RRef):
              #   logits = logits.local_value()
            loss = F.cross_entropy(logits, targets, reduction='sum')
            num_examples += targets.size(0)
            curr_loss += loss

        curr_loss = curr_loss / num_examples
        return curr_loss


def compute_epoch_metrics(model, data_loader, device, num_classes):
    model.eval()
    curr_loss, num_observations = 0., 0

    # Initialize torchmetrics
    accuracy_metric = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes).to(device)
    precision_metric = torchmetrics.Precision(task='multiclass', average='macro', num_classes=num_classes).to(device)
    recall_metric = torchmetrics.Recall(task='multiclass', average='macro', num_classes=num_classes).to(device)
    f1_metric = torchmetrics.F1Score(task='multiclass', average='macro', num_classes=num_classes).to(device)

    with torch.inference_mode():
        for features, targets in data_loader:
            features = features.to(device)
            targets = targets.to(device)
            logits, probas = model(features)

            loss = F.cross_entropy(logits, targets, reduction='sum')
            num_observations += targets.size(0)
            curr_loss += loss

            # Update metrics
            _, predicted_labels = torch.max(probas, 1)
            accuracy_metric.update(predicted_labels, targets)
            precision_metric.update(predicted_labels, targets)
            recall_metric.update(predicted_labels, targets)
            f1_metric.update(predicted_labels, targets)

        curr_loss = curr_loss / num_observations
        accuracy = (accuracy_metric.compute()).item()
        precision = (precision_metric.compute()).item()
        recall = (recall_metric.compute()).item()
        f1 = (f1_metric.compute()).item()

    return curr_loss, accuracy, precision, recall, f1

def get_model_path(model_dir, model_name):
    # Create the model directory if it does not exist
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Save the model without overwriting an existing file
    base_filename = f"{model_name}.pt"
    filename = os.path.join(model_dir, base_filename)
    counter = 1

    while os.path.exists(filename):
        filename = os.path.join(model_dir, f"{model_name}_{counter}.pt")
        counter += 1

    return filename

# plot accuracy
def plot_accuracy(log_dict):
    """function to plot train vs validation accuracy"""
    plt.plot(log_dict['train_acc_per_epoch'], label='train')
    # if log_dict['valid_acc_per_epoch'] is not an empty list
    if log_dict['valid_acc_per_epoch']:
        plt.plot(log_dict['valid_acc_per_epoch'], label='validation')
    # if log_dict['test_acc_per_epoch'] is not an empty list
    if log_dict['test_acc_per_epoch']:
        plt.plot(log_dict['test_acc_per_epoch'], label='test')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()
    return None

# plot loss
def plot_loss(log_dict):
    """
    :param log_dict:
    :return:
    """
    plt.plot(log_dict['train_loss_per_epoch'], label='train')
    # if log_dict['valid_loss_per_epoch'] is not an empty list
    if log_dict['valid_loss_per_epoch']:
        plt.plot(log_dict['valid_loss_per_epoch'], label='validation')
    # if log_dict['test_loss_per_epoch'] is not an empty list
    if log_dict['test_loss_per_epoch']:
        plt.plot(log_dict['test_loss_per_epoch'], label='test')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()
    return None

# plot learning rate
def plot_learning_rate(log_dict):
    """
    :param log_dict:
    :return:
    """
    plt.plot(log_dict['learning_rate_per_epoch'])
    plt.ylabel('Learning rate')
    plt.xlabel('Epoch')
    plt.show()
    return None