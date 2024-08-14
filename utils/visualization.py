# File: utils/visualization.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import cv2
from utils.logger import logger
import torch.nn as nn


def plot_learning_curves(train_losses, test_losses, test_accuracies, title, output_dir):
    """
    Plot learning curves showing training loss, test loss, and test accuracy over epochs.

    Args:
    - train_losses (list): List of training losses for each epoch
    - test_losses (list): List of test losses for each epoch
    - test_accuracies (list): List of test accuracies for each epoch
    - title (str): Title for the plot
    - output_dir (str): Directory to save the plot
    """
    plt.figure(figsize=(12, 8))
    epochs = range(1, len(train_losses) + 1)

    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, test_losses, 'r-', label='Test Loss')
    plt.plot(epochs, test_accuracies, 'g-', label='Test Accuracy')

    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss / Accuracy')
    plt.legend()
    plt.grid(True)

    plt.savefig(f"{output_dir}/{title.replace(' ', '_')}.png")
    plt.close()
    logger.info(f"Learning curves saved as {output_dir}/{title.replace(' ', '_')}.png")

def plot_weight_heatmaps(model, title, output_dir):
    """
    Plot heatmaps of the weight matrices for convolutional and fully connected layers.

    Args:
    - model (nn.Module): The neural network model
    - title (str): Title for the plot
    - output_dir (str): Directory to save the plot
    """
    plt.figure(figsize=(20, 15))
    
    # Plot conv1 weights
    conv1_weights = model.conv1.weight.detach().cpu().numpy()
    plt.subplot(2, 2, 1)
    sns.heatmap(conv1_weights.reshape(32, -1), cmap='viridis')
    plt.title('Conv1 Weights')

    # Plot conv2 weights
    conv2_weights = model.conv2.weight.detach().cpu().numpy()
    plt.subplot(2, 2, 2)
    sns.heatmap(conv2_weights.reshape(64, -1), cmap='viridis')
    plt.title('Conv2 Weights')

    # Plot fc1 weights
    if isinstance(model.fc1, nn.Linear):
        fc1_weights = model.fc1.weight.detach().cpu().numpy()
    else:  # LoRA layer
        fc1_weights = model.fc1.linear.weight.detach().cpu().numpy()
    plt.subplot(2, 2, 3)
    sns.heatmap(fc1_weights, cmap='viridis')
    plt.title('FC1 Weights')

    # Plot fc2 weights
    if isinstance(model.fc2, nn.Linear):
        fc2_weights = model.fc2.weight.detach().cpu().numpy()
    else:  # LoRA layer
        fc2_weights = model.fc2.linear.weight.detach().cpu().numpy()
    plt.subplot(2, 2, 4)
    sns.heatmap(fc2_weights, cmap='viridis')
    plt.title('FC2 Weights')

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{title.replace(' ', '_')}.png")
    plt.close()
    logger.info(f"Weight heatmaps saved as {output_dir}/{title.replace(' ', '_')}.png")

def plot_lora_weights(model, title, output_dir):
    """
    Plot heatmaps of the LoRA weight matrices for fully connected layers.

    Args:
    - model (nn.Module): The neural network model with LoRA layers
    - title (str): Title for the plot
    - output_dir (str): Directory to save the plot
    """
    fig, axs = plt.subplots(2, 2, figsize=(20, 15))
    fig.suptitle(title)

    def plot_lora_layer(layer, row):
        if hasattr(layer, 'lora_A'):
            axs[row, 0].imshow(layer.lora_A.detach().cpu().numpy(), cmap='viridis', aspect='auto')
            axs[row, 0].set_title(f'FC{row+1} LoRA A')
            axs[row, 1].imshow(layer.lora_B.detach().cpu().numpy(), cmap='viridis', aspect='auto')
            axs[row, 1].set_title(f'FC{row+1} LoRA B')
        elif hasattr(layer, 'lora_As'):
            combined_A = torch.cat([param.unsqueeze(0) for param in layer.lora_As], dim=0).detach().cpu().numpy()
            combined_B = torch.cat([param.unsqueeze(1) for param in layer.lora_Bs], dim=1).detach().cpu().numpy()
            axs[row, 0].imshow(combined_A, cmap='viridis', aspect='auto')
            axs[row, 0].set_title(f'FC{row+1} Combined LoRA As')
            axs[row, 1].imshow(combined_B, cmap='viridis', aspect='auto')
            axs[row, 1].set_title(f'FC{row+1} Combined LoRA Bs')
        else:
            axs[row, 0].text(0.5, 0.5, 'No LoRA weights', ha='center', va='center')
            axs[row, 1].text(0.5, 0.5, 'No LoRA weights', ha='center', va='center')

    plot_lora_layer(model.fc1, 0)
    plot_lora_layer(model.fc2, 1)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/{title.replace(' ', '_')}.png")
    plt.close()
    logger.info(f"LoRA weights heatmap saved as {output_dir}/{title.replace(' ', '_')}.png")

def visualize_feature_space(model, device, data_loader, title, output_dir):
    """
    Visualize the feature space of the model using t-SNE.

    Args:
    - model (nn.Module): The trained neural network model
    - device (torch.device): The device to run the model on
    - data_loader (DataLoader): DataLoader for the dataset to visualize
    - title (str): Title for the plot
    - output_dir (str): Directory to save the plot
    """
    model.eval()
    features = []
    labels = []

    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device)
            # Forward pass through convolutional layers
            x = model.conv1(data)
            x = torch.nn.functional.relu(x)
            x = model.conv2(x)
            x = torch.nn.functional.relu(x)
            x = torch.nn.functional.max_pool2d(x, 2)
            x = model.dropout1(x)
            # Flatten the output
            x = torch.flatten(x, 1)
            # Forward pass through fc1
            feature = model.fc1(x)
            features.append(feature.cpu().numpy())
            labels.extend(target.numpy())

    features = np.concatenate(features)
    
    tsne = TSNE(n_components=2, random_state=42)
    features_tsne = tsne.fit_transform(features)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features_tsne[:, 0], features_tsne[:, 1], c=labels, cmap='tab10')
    plt.colorbar(scatter)
    plt.title(title)
    plt.savefig(f"{output_dir}/{title.replace(' ', '_')}.png")
    plt.close()
    logger.info(f"Feature space visualization saved as {output_dir}/{title.replace(' ', '_')}.png")

def plot_confusion_matrix(model, device, data_loader, class_names, title, output_dir):
    """
    Generate and plot a confusion matrix for the model's predictions.

    Args:
    - model (nn.Module): The trained neural network model
    - device (torch.device): The device to run the model on
    - data_loader (DataLoader): DataLoader for the test dataset
    - class_names (list): List of class names
    - title (str): Title for the plot
    - output_dir (str): Directory to save the plot
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, preds = torch.max(output, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(f"{output_dir}/{title.replace(' ', '_')}.png")
    plt.close()
    logger.info(f"Confusion matrix saved as {output_dir}/{title.replace(' ', '_')}.png")

def grad_cam(model, device, image, target_class, layer_name='conv2'):
    """
    Generate a Grad-CAM heatmap for a given image and target class.

    Args:
    - model (nn.Module): The trained neural network model
    - device (torch.device): The device to run the model on
    - image (torch.Tensor): Input image tensor
    - target_class (int): Target class for Grad-CAM
    - layer_name (str): Name of the layer to use for Grad-CAM

    Returns:
    - numpy.ndarray: Grad-CAM heatmap
    """
    model.eval()
    image = image.unsqueeze(0).to(device)
    image.requires_grad_()

    # Get the output for the target class
    output = model(image)
    model.zero_grad()
    class_output = output[0, target_class]
    class_output.backward()

    # Get the gradients and activations
    gradients = getattr(model, layer_name).weight.grad.data.cpu().numpy()
    activations = getattr(model, layer_name)(image).detach().cpu().numpy()

    # Calculate Grad-CAM
    weights = np.mean(gradients, axis=(2, 3))[0, :]
    cam = np.sum(weights[:, np.newaxis, np.newaxis] * activations[0], axis=0)
    cam = np.maximum(cam, 0)  # ReLU
    cam = cv2.resize(cam, (28, 28))  # Resize to image size
    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize

    return cam

def plot_grad_cam(model, device, image, target_class, title, output_dir):
    """
    Generate and plot Grad-CAM heatmap for a given image.

    Args:
    - model (nn.Module): The trained neural network model
    - device (torch.device): The device to run the model on
    - image (torch.Tensor): Input image tensor
    - target_class (int): Target class for Grad-CAM
    - title (str): Title for the plot
    - output_dir (str): Directory to save the plot
    """
    cam = grad_cam(model, device, image, target_class)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(image.squeeze().cpu().numpy(), cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(cam, cmap='jet')
    plt.title('Grad-CAM Heatmap')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(image.squeeze().cpu().numpy(), cmap='gray')
    plt.imshow(cam, cmap='jet', alpha=0.5)
    plt.title('Overlay')
    plt.axis('off')

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{title.replace(' ', '_')}.png")
    plt.close()
    logger.info(f"Grad-CAM visualization saved as {output_dir}/{title.replace(' ', '_')}.png")