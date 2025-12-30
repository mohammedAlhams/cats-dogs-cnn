import random
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def binary_accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    """
    outputs: logits (before sigmoid), shape (N, 1)
    labels: 0 or 1, shape (N,)
    """
    preds = torch.sigmoid(outputs)
    preds = (preds >= 0.5).float().view(-1)
    labels = labels.float().view(-1)
    correct = (preds == labels).sum().item()
    return correct / labels.size(0)


def train_one_epoch(model, loader: DataLoader, criterion, optimizer, device) -> Dict:
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs.view(-1), labels.float())
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        running_acc += binary_accuracy(outputs, labels) * batch_size
        total += batch_size

    epoch_loss = running_loss / total
    epoch_acc = running_acc / total
    return {"loss": epoch_loss, "acc": epoch_acc}


def evaluate(model, loader: DataLoader, criterion, device) -> Dict:
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs.view(-1), labels.float())

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            running_acc += binary_accuracy(outputs, labels) * batch_size
            total += batch_size

    epoch_loss = running_loss / total
    epoch_acc = running_acc / total
    return {"loss": epoch_loss, "acc": epoch_acc}


def plot_curves(train_losses: List[float], val_losses: List[float],
                train_accs: List[float], val_accs: List[float],
                prefix: str = "baseline"):
    epochs = range(1, len(train_losses) + 1)

    # Loss
    plt.figure()
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{prefix} Loss")
    plt.legend()
    plt.savefig(f"{prefix}_loss.png", bbox_inches="tight")
    plt.close()

    # Accuracy
    plt.figure()
    plt.plot(epochs, train_accs, label="Train Acc")
    plt.plot(epochs, val_accs, label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{prefix} Accuracy")
    plt.legend()
    plt.savefig(f"{prefix}_accuracy.png", bbox_inches="tight")
    plt.close()
