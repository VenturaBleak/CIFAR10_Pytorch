from metrics import compute_accuracy, compute_epoch_loss

import time
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

def train_classifier_simple_v1(num_epochs, model,
                               optimizer, device,
                               train_loader_aug,
                               train_loader,
                               test_loader,
                               valid_loader=None,
                               scheduler=None,
                               loss_fn=None,
                               logging_interval=100,
                               skip_epoch_stats=False):
    """
    Train a PyTorch classifier model with a simple training loop.

    Args:
        num_epochs (int): The number of epochs to train the model for.
        model (nn.Module): The PyTorch model to train.
        optimizer (torch.optim.Optimizer): The optimizer used to update the model parameters.
        device (str): The device used to train the model ('cpu' or 'cuda').
        train_loader (DataLoader): The PyTorch DataLoader for the training data.
        test_loader (DataLoader): The PyTorch DataLoader for the test data.
        valid_loader (DataLoader, optional): The PyTorch DataLoader for the validation data. Default: None.
        loss_fn (callable, optional): The loss function used to compute the loss. Default: None (cross entropy loss).
        logging_interval (int, optional): The interval at which to print the training loss. Default: 100.
        skip_epoch_stats (bool, optional): Whether to skip computing and logging epoch statistics. Default: False.

    Returns:
        dict: A dictionary containing the training and validation statistics, including
            train_loss_per_batch, train_acc_per_epoch, train_loss_per_epoch, valid_acc_per_epoch,
            and valid_loss_per_epoch.
    """
    # logging
    log_dict = {'train_loss_per_batch': [],
                'train_acc_per_epoch': [],
                'train_loss_per_epoch': [],
                'valid_acc_per_epoch': [],
                'valid_loss_per_epoch': [],
                'test_acc_per_epoch': [],
                'test_loss_per_epoch': [],
                'learning_rate_per_epoch': []}

    # cross entropy loss per default
    if loss_fn is None:
        loss_fn = F.cross_entropy

    start_time = time.time()
    for epoch in range(num_epochs):

        model.train()
        for batch_idx, (features, targets) in enumerate(train_loader_aug):

            features = features.to(device)
            targets = targets.to(device)

            # FORWARD AND BACK PROP
            logits, probas = model(features)

            # checks if the logits tensor is a remote reference object (RRef) created by PyTorch's distributed RPC framework
            # This is necessary because remote reference objects are created by the RPC framework to store tensor data
            # across different processes, and the actual tensor values are not immediately available.
            # if isinstance(logits, torch.distributed.rpc.api.RRef):
              #   logits = logits.local_value()
            loss = loss_fn(logits, targets)
            optimizer.zero_grad()

            loss.backward()

            # UPDATE MODEL PARAMETERS
            optimizer.step()

            # LOGGING
            log_dict['train_loss_per_batch'].append(loss.item())

            if batch_idx % max(1,int(len(train_loader_aug)*(1/5))) == 0:
                print(f'Epoch: {epoch + 1:03d}/{num_epochs:03d} | '
                      f'Batch {batch_idx + 1:03d}/{len(train_loader_aug):03d} |'
                      f' Loss: {loss:.4f}')

        if scheduler is not None:
            # logging learning rate
            log_dict['learning_rate_per_epoch'].append(optimizer.param_groups[0]['lr'])
            scheduler.step()

        if not skip_epoch_stats:
            model.eval()

            with torch.inference_mode():  # save memory during inference
                train_acc = compute_accuracy(model, train_loader, device)
                train_loss = compute_epoch_loss(model, train_loader, device)
                print('***Epoch: %03d/%03d | Train. Acc.: %.3f%% | Train Loss: %.3f' % (
                    epoch + 1, num_epochs, train_acc, train_loss))
                log_dict['train_loss_per_epoch'].append(train_loss.item())
                log_dict['train_acc_per_epoch'].append(train_acc.item())

                if valid_loader is not None:
                    valid_acc = compute_accuracy(model, valid_loader, device)
                    valid_loss = compute_epoch_loss(model, valid_loader, device)
                    print('***Epoch: %03d/%03d | Valid. Acc.: %.3f%% | Valid. Loss: %.3f' % (
                        epoch + 1, num_epochs, valid_acc, valid_loss))
                    log_dict['valid_loss_per_epoch'].append(valid_loss.item())
                    log_dict['valid_acc_per_epoch'].append(valid_acc.item())

                test_acc = compute_accuracy(model, test_loader, device)
                test_loss = compute_epoch_loss(model, test_loader, device)
                print('***Epoch: %03d/%03d | Test Acc.: %.3f%% | Test Loss: %.3f' % (
                    epoch + 1, num_epochs, test_acc, test_loss))
                log_dict['test_loss_per_epoch'].append(test_loss.item())
                log_dict['test_acc_per_epoch'].append(test_acc.item())

        print('Time elapsed: %.2f min' % ((time.time() - start_time) / 60))

    print('Total Training Time: %.2f min' % ((time.time() - start_time) / 60))

    return log_dict

def train_classifier_simple_v2(
        model, num_epochs, train_loader_aug, train_loader,
        optimizer,
        device,
        valid_loader=None,
        test_loader=None,
        loss_fn=None,
        best_model_save_path=None,
        scheduler=None,
        skip_train_acc=False):
    # either only valid or only test loader, raise error if both are None
    if valid_loader is None and test_loader is None:
        raise ValueError('Either valid_loader or test_loader must be provided.')
    if valid_loader is not None and test_loader is not None:
        raise ValueError('Only one of valid_loader or test_loader can be provided, not both.')

    start_time = time.time()
    log_dict = {'train_loss_per_batch': [],
                'train_acc_per_epoch': [],
                'train_loss_per_epoch': [],
                'valid_acc_per_epoch': [],
                'valid_loss_per_epoch': [],
                'test_acc_per_epoch': [],
                'test_loss_per_epoch': [],
                'learning_rate_per_epoch': []}

    # cross entropy loss per default
    if loss_fn is None:
        loss_fn = F.cross_entropy

    best_valid_acc, best_test_acc, best_epoch = -float('inf'), -float('inf'), 0

    for epoch in tqdm(range(num_epochs)):

        model.train()
        for batch_idx, (features, targets) in enumerate(train_loader_aug):

            features = features.to(device)
            targets = targets.to(device)

            # ## FORWARD AND BACK PROP
            logits, probas = model(features)
            loss = loss_fn(logits, targets)
            optimizer.zero_grad()

            loss.backward()

            # ## UPDATE MODEL PARAMETERS
            optimizer.step()

            # ## LOGGING
            # log_dict['train_loss_per_batch'].append(loss.item())

            # if batch_idx % max(1,int(len(train_loader_aug)*(1/5))) == 0:
            #     print(f'Epoch: {epoch + 1:03d}/{num_epochs:03d} | '
            #           f'Batch {batch_idx + 1:03d}/{len(train_loader_aug):03d} |'
            #           f' Loss: {loss:.4f}')

        if scheduler is not None:
            # logging learning rate
            log_dict['learning_rate_per_epoch'].append(optimizer.param_groups[0]['lr'])
            scheduler.step()

        model.eval()
        with torch.inference_mode():  # save memory during inference
            # compute train accuracy
            if not skip_train_acc:
                train_acc = compute_accuracy(model, train_loader, device=device)
            else:
                train_acc = float('nan')
            train_loss = compute_epoch_loss(model, train_loader, device)
            log_dict['train_loss_per_epoch'].append(train_loss.item())
            log_dict['train_acc_per_epoch'].append(train_acc.item())

            if valid_loader is not None:
                # compute validation accuracy
                valid_acc = compute_accuracy(model, valid_loader, device)
                valid_loss = compute_epoch_loss(model, valid_loader, device)
                log_dict['valid_loss_per_epoch'].append(valid_loss.item())
                log_dict['valid_acc_per_epoch'].append(valid_acc.item())

                if valid_acc.item() > best_valid_acc:
                    best_valid_acc, best_epoch = valid_acc.item(), epoch+1
                    if best_model_save_path:
                        torch.save(model.state_dict(), best_model_save_path)

                print(f'***Epoch: {epoch + 1:03d}/{num_epochs:03d} '
                      f'| Train Loss: {train_loss :.4f} '
                      f'| Valid. Loss: {valid_loss :.4f} '
                      f'| Train Acc: {train_acc :.2f}% '
                      f'| Valid. Acc: {valid_acc :.2f}% '
                      f'| Best Validation '
                      f'(Ep. {best_epoch:03d}): {best_valid_acc :.2f}%')

            # compute test accuracy
            if test_loader is not None:
                test_acc = compute_accuracy(model, test_loader, device)
                test_loss = compute_epoch_loss(model, test_loader, device)
                log_dict['test_loss_per_epoch'].append(test_loss.item())
                log_dict['test_acc_per_epoch'].append(test_acc.item())

                if test_acc.item() > best_test_acc:
                    best_test_acc, best_epoch = test_acc.item(), epoch+1
                    if best_model_save_path:
                        torch.save(model.state_dict(), best_model_save_path)

                print(f'***Epoch: {epoch + 1:03d}/{num_epochs:03d} '
                      f'| Train Loss: {train_loss :.4f} '
                      f'| Test Loss: {test_loss :.4f} '
                      f'| Train Acc: {train_acc :.2f}% '
                      f'| Test Acc: {test_acc :.2f}% '
                      f'| Best Test '
                      f'(Ep. {best_epoch:03d}): {best_test_acc :.2f}%')

    elapsed = (time.time() - start_time)/60
    print(f'Total Training Time Elapsed: {elapsed:.2f} min')

    return log_dict