import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from .models import DenseModel, CNNModel

# --- UTILITAIRES ---


def calculate_accuracy(outputs, targets):
    _, preds = torch.max(outputs, 1)
    return (preds == targets).float().mean().item()