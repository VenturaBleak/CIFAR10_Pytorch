import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def compute_accuracy(model, data_loader, device):
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
        return correct_pred.float() / num_examples * 100

def compute_epoch_loss(model, data_loader, device):
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