# File: src/train.py

import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils.logger import logger

def train(model, device, train_loader, optimizer, epoch):
    """
    Train the model for one epoch.

    Args:
    - model (nn.Module): The neural network model to train
    - device (torch.device): The device to run the model on
    - train_loader (DataLoader): DataLoader for the training dataset
    - optimizer (torch.optim.Optimizer): The optimizer for updating model parameters
    - epoch (int): The current epoch number

    Returns:
    - float: Average loss for this epoch
    """
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if batch_idx % 100 == 0:
            logger.info(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

    avg_loss = total_loss / len(train_loader)
    logger.info(f'Train Epoch: {epoch}, Average loss: {avg_loss:.4f}')
    return avg_loss

def test(model, device, test_loader):
    """
    Test the model on the test dataset.

    Args:
    - model (nn.Module): The neural network model to test
    - device (torch.device): The device to run the model on
    - test_loader (DataLoader): DataLoader for the test dataset

    Returns:
    - float: Average loss on the test set
    - float: Accuracy on the test set
    """
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="Testing"):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    logger.info(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({accuracy:.2f}%)')
    
    return test_loss, accuracy

def fine_tune(model, device, train_loader, optimizer, epoch):
    """
    Fine-tune the model for one epoch.

    Args:
    - model (nn.Module): The neural network model to fine-tune
    - device (torch.device): The device to run the model on
    - train_loader (DataLoader): DataLoader for the fine-tuning dataset
    - optimizer (torch.optim.Optimizer): The optimizer for updating model parameters
    - epoch (int): The current epoch number

    Returns:
    - float: Average loss for this epoch
    """
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f"Fine-tune Epoch {epoch}")):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if batch_idx % 100 == 0:
            logger.info(f'Fine-tune Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

    avg_loss = total_loss / len(train_loader)
    logger.info(f'Fine-tune Epoch: {epoch}, Average loss: {avg_loss:.4f}')
    return avg_loss

def train_and_evaluate(model, device, train_loader, test_loader, optimizer, num_epochs):
    """
    Train and evaluate the model for a specified number of epochs.

    Args:
    - model (nn.Module): The neural network model to train and evaluate
    - device (torch.device): The device to run the model on
    - train_loader (DataLoader): DataLoader for the training dataset
    - test_loader (DataLoader): DataLoader for the test dataset
    - optimizer (torch.optim.Optimizer): The optimizer for updating model parameters
    - num_epochs (int): Number of epochs to train for

    Returns:
    - list: Training losses for each epoch
    - list: Test losses for each epoch
    - list: Test accuracies for each epoch
    """
    train_losses = []
    test_losses = []
    test_accuracies = []

    for epoch in range(1, num_epochs + 1):
        train_loss = train(model, device, train_loader, optimizer, epoch)
        test_loss, test_accuracy = test(model, device, test_loader)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

    return train_losses, test_losses, test_accuracies