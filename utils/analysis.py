# File: utils/analysis.py

import torch
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.train import train, test
from utils.logger import logger
import itertools

def hyperparameter_tuning(model_class, device, train_loader, val_loader, param_grid, epochs):
    """
    Perform grid search for hyperparameter tuning.

    Args:
    - model_class: The model class to instantiate
    - device: The device to run on (cpu or cuda)
    - train_loader: DataLoader for training data
    - val_loader: DataLoader for validation data
    - param_grid: Dictionary of hyperparameters to try
    - epochs: Number of epochs to train for each combination

    Returns:
    - best_params: Dictionary of best hyperparameters
    - best_accuracy: Best validation accuracy achieved
    """
    best_accuracy = 0
    best_params = {}

    for params in itertools.product(*param_grid.values()):
        current_params = dict(zip(param_grid.keys(), params))
        logger.info(f"Trying parameters: {current_params}")

        model = model_class().to(device)
        optimizer = optim.Adam(model.parameters(), lr=current_params['lr'])

        for epoch in range(epochs):
            train(model, device, train_loader, optimizer, epoch)

        _, accuracy = test(model, device, val_loader)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = current_params

    return best_params, best_accuracy

def compute_model_size(model):
    """
    Compute the size of the model in terms of number of parameters.

    Args:
    - model: PyTorch model

    Returns:
    - total_params: Total number of parameters
    - trainable_params: Number of trainable parameters
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def evaluate_model(model, device, test_loader):
    """
    Evaluate the model and compute various metrics.

    Args:
    - model: PyTorch model
    - device: The device to run on (cpu or cuda)
    - test_loader: DataLoader for test data

    Returns:
    - metrics: Dictionary containing various evaluation metrics
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            all_preds.extend(pred.view(-1).cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    metrics = {
        'accuracy': accuracy_score(all_targets, all_preds),
        'precision': precision_score(all_targets, all_preds, average='weighted'),
        'recall': recall_score(all_targets, all_preds, average='weighted'),
        'f1_score': f1_score(all_targets, all_preds, average='weighted')
    }

    return metrics

def compare_models(models, device, test_loader):
    """
    Compare multiple models on the same test set.

    Args:
    - models: Dictionary of models to compare (name: model)
    - device: The device to run on (cpu or cuda)
    - test_loader: DataLoader for test data

    Returns:
    - comparison: Dictionary containing comparison results
    """
    comparison = {}
    for name, model in models.items():
        metrics = evaluate_model(model, device, test_loader)
        size, trainable_size = compute_model_size(model)
        comparison[name] = {
            'metrics': metrics,
            'total_params': size,
            'trainable_params': trainable_size
        }
    return comparison