import torch.nn as nn
import torch.nn.functional as F


class BaselineCNN(nn.Module):
    """
    Model 1: Simple CNN for binary classification (cat vs dog).
    Small, fast, easy to explain.
    """
    def __init__(self):
        super(BaselineCNN, self).__init__()
        # Input: 3 x 128 x 128 (we'll resize images to 128x128)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)   # -> 16 x 128 x 128
        self.pool1 = nn.MaxPool2d(2, 2)                           # -> 16 x 64 x 64

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # -> 32 x 64 x 64
        self.pool2 = nn.MaxPool2d(2, 2)                           # -> 32 x 32 x 32

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # -> 64 x 32 x 32
        self.pool3 = nn.MaxPool2d(2, 2)                           # -> 64 x 16 x 16

        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 1)  # output: 1 logit for binary classification

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # no sigmoid here, we'll use BCEWithLogitsLoss
        return x


class ImprovedCNN(nn.Module):
    """
    Model 2: Deeper CNN with BatchNorm + Dropout.
    This is your 'significantly modified' architecture.
    """
    def __init__(self):
        super(ImprovedCNN, self).__init__()

        # Block 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)  # -> 32 x 64 x 64

        # Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)  # -> 64 x 32 x 32

        # Block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)  # -> 128 x 16 x 16

        self.dropout = nn.Dropout(0.5)

        self.fc1 = nn.Linear(128 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, 1)  # 1 logit, binary

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))

        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
