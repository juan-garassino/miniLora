# File: src/data.py

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from utils.logger import logger
import os
from PIL import Image

def load_mnist(batch_size=64):
    """
    Load the MNIST dataset.

    Args:
    - batch_size (int): The batch size for DataLoader. Default is 64.

    Returns:
    - train_loader (DataLoader): DataLoader for the training set
    - test_loader (DataLoader): DataLoader for the test set
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('data', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    logger.info(f"MNIST dataset loaded. Training set size: {len(train_dataset)}, Test set size: {len(test_dataset)}")

    return train_loader, test_loader

def load_fashion_mnist(batch_size=64):
    """
    Load the Fashion MNIST dataset.

    Args:
    - batch_size (int): The batch size for DataLoader. Default is 64.

    Returns:
    - train_loader (DataLoader): DataLoader for the training set
    - test_loader (DataLoader): DataLoader for the test set
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])

    train_dataset = datasets.FashionMNIST('data', train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST('data', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    logger.info(f"Fashion MNIST dataset loaded. Training set size: {len(train_dataset)}, Test set size: {len(test_dataset)}")

    return train_loader, test_loader

class CustomDataset(Dataset):
    """
    Custom dataset that supports loading images from a directory.
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = [f for f in os.listdir(root_dir) if f.endswith('.png') or f.endswith('.jpg')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.images[idx])
        image = Image.open(img_name).convert('L')  # Convert to grayscale
        
        if self.transform:
            image = self.transform(image)
        
        return image, 0  # Return 0 as label (you may want to modify this for your use case)

def load_custom_dataset(data_dir, batch_size=64):
    """
    Load a custom dataset from a directory.

    Args:
    - data_dir (str): Path to the directory containing the dataset
    - batch_size (int): The batch size for DataLoader. Default is 64.

    Returns:
    - data_loader (DataLoader): DataLoader for the custom dataset
    """
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = CustomDataset(data_dir, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    logger.info(f"Custom dataset loaded from {data_dir}. Dataset size: {len(dataset)}")

    return data_loader

# You can add more data loading functions or data augmentation techniques here