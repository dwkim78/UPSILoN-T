import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset


class GenesisNet(nn.Module):
    """Convolution neural network for UPSILoN-T."""

    def __init__(self, n_final: int=19):
        """
        Defines network structures.

        Args:
            n_final: The number of features of the final layers.
                For the pre-trained model, 19. For the transferred model,
                it is the number of classes of the target datasets.
        """

        super(Net, self).__init__()

        # Three convolution layers.
        self.conv1 = nn.Conv2d(1, 9, 2, padding=1)
        self.conv2 = nn.Conv2d(9, 18, 2, padding=1)
        self.conv3 = nn.Conv2d(18, 36, 2, padding=1)

        # Three batch normal and three fully-connected layers.
        # For 4x4 matrix input shape.
        self.bn0 = nn.BatchNorm1d(36 * 7 * 7)
        self.fc1 = nn.Linear(36 * 7 * 7, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, n_final)
        self.bn3 = nn.BatchNorm1d(n_final)

    def forward(self, x):
        """Forward module."""
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)

        # print(x.shape)
        # For 4x4 matrix input shape.
        x = x.view(-1, 36 * 7 * 7)
        x = self.bn0(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.bn1(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.bn2(x)
        x = self.fc3(x)
        x = self.bn3(x)

        return x


class CnnNet(nn.Module):
    """Convolution neural network for UPSILoN-T."""

    def __init__(self, n_final: int=19):
        """
        Defines network structures.

        Args:
            n_final: The number of features of the final layers.
                For the pre-trained model, 19. For the transferred model,
                it is the number of classes of the target datasets.
        """

        super(Net, self).__init__()

        # Three convolution layers.
        # self.conv1 = nn.Conv2d(1, 64, 2, padding=1)
        # self.conv2 = nn.Conv2d(64, 128, 2, padding=1)
        # self.conv3 = nn.Conv2d(128, 256, 2, padding=1)

        self.conv1 = nn.Conv2d(1, 9, 2, padding=1)
        self.conv2 = nn.Conv2d(9, 18, 2, padding=1)
        self.conv3 = nn.Conv2d(18, 36, 2, padding=1)

        # self.conv1 = nn.Conv2d(1, 9, (1, 4), padding=1)
        # self.conv2 = nn.Conv2d(9, 18, (1, 4), padding=1)
        # self.conv3 = nn.Conv2d(18, 36, (1, 4), padding=1)

        # self.conv1 = nn.Conv2d(1, 9, (1, 2), padding=1)
        # self.conv2 = nn.Conv2d(9, 18, (1, 2), padding=1)
        # self.conv3 = nn.Conv2d(18, 36, (1, 2), padding=1)

        # Three batch normal and three fully-connected layers.
        # For 4x4 matrix input shape.
        self.bn0 = nn.BatchNorm1d(36 * 7 * 7)
        self.fc1 = nn.Linear(36 * 7 * 7, 256)
        # For 1x16 matrix input shape. 2x2 kernel. Test usage only.
        # self.bn0 = nn.BatchNorm1d(36 * 4 * 19)
        # self.fc1 = nn.Linear(36 * 4 * 19, 256)
        # For 1x16 matrix input shape. 1x4 kernel. Test usage only.
        # self.bn0 = nn.BatchNorm1d(36 * 7 * 13)
        # self.fc1 = nn.Linear(36 * 7 * 13, 256)
        # For 1x16 matrix input shape. 1x4 kernel. Test usage only.
        # self.bn0 = nn.BatchNorm1d(36 * 7 * 19)
        # self.fc1 = nn.Linear(36 * 7 * 19, 256)

        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, n_final)
        self.bn3 = nn.BatchNorm1d(n_final)

    def forward(self, x):
        """Forward module."""
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)

        # print(x.shape)
        # For 4x4 matrix input shape.
        x = x.view(-1, 36 * 7 * 7)
        # For 1x16 matrix input shape. 2x2 kernel. Test usage only.
        # x = x.view(-1, 36 * 4 * 19)
        # For 1x16 matrix input shape. 1x4 kernel. Test usage only.
        # x = x.view(-1, 36 * 7 * 13)
        # For 1x16 matrix input shape. 1x2 kernel. Test usage only.
        # x = x.view(-1, 36 * 7 * 19)

        x = self.bn0(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.bn1(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.bn2(x)
        x = self.fc3(x)
        x = self.bn3(x)

        return x


class Net(nn.Module):
    """Deep neural network for UPSILoN-T."""

    def __init__(self, n_final: int=19):
        """
        Defines network structures.

        Args:
            n_final: The number of features of the final layers.
                For the pre-trained model, 19. For the transferred model,
                it is the number of classes of the target datasets.
        """

        super(Net, self).__init__()

        # Three batch normal and three fully-connected layers.
        self.bn0 = nn.BatchNorm1d(16, 32)
        self.fc1 = nn.Linear(16, 32)
        self.bn1 = nn.BatchNorm1d(32)
        # self.fc2 = nn.Linear(32, 64)
        # self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(32, n_final)
        self.bn3 = nn.BatchNorm1d(n_final)

    def forward(self, x):
        """Forward module."""

        # print(x.shape)
        # For 4x4 matrix input shape.
        x = x.view(-1, 16)

        x = self.bn0(x)
        x = self.fc1(x)
        x = F.relu(x)
        # x = self.bn1(x)
        # x = self.fc2(x)
        # x = F.relu(x)
        # x = self.bn2(x)
        x = self.fc3(x)
        x = self.bn3(x)

        return x


class LightCurveDataset(Dataset):
    """Lightcurve dataset"""

    def __init__(self, features: np.ndarray, labels: np.ndarray, ids=None):
        """
        Returns sampled data.

        Args:
            features: Numpy array of features.
            labels: Numpy array of labels.
            ids: (optional) Numpy array of each sample's ID.

        Returns:
            sampled features, sampled labels, sampled IDs (optional)
        """

        # Convert numpy array to PyTorch tensor.
        self.features = torch.from_numpy(features).float()
        self.labels = torch.from_numpy(labels).long()
        self.ids = ids

    def __len__(self):
        """Returns the length of dataset."""
        return len(self.labels)

    def __getitem__(self, idx):
        """Returns sampled data."""

        # Reshape into 4 by 4 matrix.
        # sample_features = self.features[idx].reshape(1, 4, 4)
        # Reshape into 1 by 16 matrix. Test usage only.
        sample_features = self.features[idx].reshape(1, 1, 16)
        sample_labels = self.labels[idx]
        if self.ids is not None:
            sample_ids = self.ids[idx]

        if self.ids is not None:
            return sample_features, sample_labels, sample_ids
        else:
            return sample_features, sample_labels


