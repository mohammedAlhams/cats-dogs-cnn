import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models import BaselineCNN, ImprovedCNN
from utils import set_seed, train_one_epoch, evaluate, plot_curves


def get_dataloaders(data_dir: str = "data", batch_size: int = 32):
    """
    Expects folders:
        data/train/cats, data/train/dogs
        data/val/cats, data/val/dogs
        data/test/cats, data/test/dogs
    """
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])

    train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"),
                                         transform=train_transform)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, "val"),
                                       transform=val_test_transform)
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, "test"),
                                        transform=val_test_transform)

    print("Class to index mapping:", train_dataset.class_to_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def train_model(model_name: str = "baseline",
                num_epochs: int = 10,
                batch_size: int = 32,
                lr: float = 1e-3,
                weight_decay: float = 0.0,
                use_scheduler: bool = False):

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = get_dataloaders(batch_size=batch_size)

    if model_name == "baseline":
        model = BaselineCNN()
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif model_name == "improved":
        model = ImprovedCNN()
        optimizer = optim.SGD(model.parameters(), lr=lr,
                              momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError("model_name must be 'baseline' or 'improved'")

    model.to(device)

    criterion = nn.BCEWithLogitsLoss()

    scheduler = None
    if use_scheduler:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(1, num_epochs + 1):
        train_stats = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_stats = evaluate(model, val_loader, criterion, device)

        if scheduler is not None:
            scheduler.step()

        train_losses.append(train_stats["loss"])
        val_losses.append(val_stats["loss"])
        train_accs.append(train_stats["acc"])
        val_accs.append(val_stats["acc"])

        print(
            f"Epoch [{epoch}/{num_epochs}] "
            f"Train Loss: {train_stats['loss']:.4f} | Train Acc: {train_stats['acc']*100:.2f}% "
            f"| Val Loss: {val_stats['loss']:.4f} | Val Acc: {val_stats['acc']*100:.2f}%"
        )

    test_stats = evaluate(model, test_loader, criterion, device)
    print(f"\n{model_name.capitalize()} Test Loss: {test_stats['loss']:.4f} | "
          f"Test Acc: {test_stats['acc']*100:.2f}%")

    plot_curves(train_losses, val_losses, train_accs, val_accs, prefix=model_name)
    torch.save(model.state_dict(), f"{model_name}_cats_dogs.pth")
    print(f"Saved weights to {model_name}_cats_dogs.pth")

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accs": train_accs,
        "val_accs": val_accs,
        "test_loss": test_stats["loss"],
        "test_acc": test_stats["acc"],
    }


if __name__ == "__main__":
    print("Training Baseline CNN...")
    baseline_results = train_model(
        model_name="baseline",
        num_epochs=8,
        batch_size=32,
        lr=1e-3,
        weight_decay=0.0,
        use_scheduler=False
    )

    print("\nTraining Improved CNN...")
    improved_results = train_model(
        model_name="improved",
        num_epochs=12,
        batch_size=32,
        lr=0.01,
        weight_decay=5e-4,
        use_scheduler=True
    )

    print("\nBaseline Test Acc: {:.2f}%".format(baseline_results["test_acc"] * 100))
    print("Improved Test Acc: {:.2f}%".format(improved_results["test_acc"] * 100))
