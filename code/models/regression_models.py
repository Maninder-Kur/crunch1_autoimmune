import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.nn import MSELoss
from torch.utils.data import ConcatDataset
from scipy.stats import pearsonr, ConstantInputWarning
from scipy.spatial.distance import cdist
from torch.nn import DataParallel
import math

class CNN_regression_model(nn.Module):
    def __init__(self, input_height=16, input_width=1024, output_size=460):
        super(CNN_regression_model, self).__init__()

        # Set kernel sizes
        kernel_size_1 = (8, 1)
        kernel_size_2 = (1, 100)
        kernel_size_3 = (3, 3)
        pool_size_1 = (1, 2)  # Max pooling size
        pool_size_2 = (1, 10)
        pool_size_3 = (2, 2)

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=kernel_size_1)
        self.pool1 = nn.MaxPool2d(pool_size_1)
        self.bn1 = nn.BatchNorm2d(128)

        self.conv2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=kernel_size_2)
        self.pool2 = nn.MaxPool2d(pool_size_2)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=kernel_size_3)
        self.pool3 = nn.MaxPool2d(pool_size_3)
        self.bn3 = nn.BatchNorm2d(64)
        # self.conv3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=kernel_size_3)

        # Compute output dimensions after convolutions and pooling
        conv1_out_height, conv1_out_width = self.compute_conv_output_size(input_height, input_width, kernel_size_1)
        # print(f"conv1 output size: ({conv1_out_height}, {conv1_out_width})")
        
        pool1_out_height, pool1_out_width = self.compute_pool_output_size(conv1_out_height, conv1_out_width, pool_size_1)
        # print(f"pool1 output size: ({pool1_out_height}, {pool1_out_width})")

        conv2_out_height, conv2_out_width = self.compute_conv_output_size(pool1_out_height, pool1_out_width, kernel_size_2)
        # print(f"conv2 output size: ({conv2_out_height}, {conv2_out_width})")
        
        pool2_out_height, pool2_out_width = self.compute_pool_output_size(conv2_out_height, conv2_out_width, pool_size_2)
        # print(f"pool2 output size: ({pool2_out_height}, {pool2_out_width})")

        conv3_out_height, conv3_out_width = self.compute_conv_output_size(pool2_out_height, pool2_out_width, kernel_size_3)
        # print(f"conv3 output size: ({conv3_out_height}, {conv3_out_width})")

        pool3_out_height, pool3_out_width = self.compute_pool_output_size(conv3_out_height, conv3_out_width, pool_size_3)

        # Dynamically calculate flattened size based on actual output dimensions after conv3
        self.flattened_size = 64 * pool3_out_height * pool3_out_width  # Update flattened size
        # print(f"Flattened size: {self.flattened_size}")

        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, 1024)  # Use calculated flattened_size
        self.bn_fc = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, output_size)

    def compute_conv_output_size(self, input_height, input_width, kernel_size, stride=1, padding=0):
        # Formula to calculate output size after convolution or pooling
        output_height = (input_height - kernel_size[0] + 2 * padding) // stride + 1
        output_width = (input_width - kernel_size[1] + 2 * padding) // stride + 1
        return output_height, output_width

    def compute_pool_output_size(self, input_height, input_width, kernel_size, padding=0):
        # Formula to calculate output size after convolution or pooling with stride=2
        output_height = (input_height - kernel_size[0] + 2 * padding) // kernel_size[0] + 1
        output_width = (input_width - kernel_size[1] + 2 * padding) // kernel_size[1] + 1
        
        return output_height, output_width

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        # print(f"Shape after conv1: {x.shape}")  # Debugging shape
        
        x = self.pool1(x)
        # print(f"Shape after pool1: {x.shape}")  # Debugging shape

        x = F.relu(self.bn2(self.conv2(x)))
        # print(f"Shape after conv2: {x.shape}")  # Debugging shape
        
        x = self.pool2(x)
        # print(f"Shape after pool2: {x.shape}")  # Debugging shape

        # x = F.relu(self.conv3(x))
        # print(f"Shape after conv3: {x.shape}")  # Debugging shape

        x = F.relu(self.bn3(self.conv3(x)))
        # print(f"Shape after conv2: {x.shape}")  # Debugging shape
        
        x = self.pool3(x)
        # print(f"Shape after pool2: {x.shape}")  # Debugging shape


        x = x.view(x.size(0), -1)  # Flatten the tensor
        # print(f"Shape after flatten: {x.shape}")  # Debugging shape

        x = F.relu(self.bn_fc(self.fc1(x)))
        # print(f"Shape after fc1: {x.shape}")  # Debugging shape
        
        x = self.fc2(x)
        return x
        
class RegressionModel(nn.Module):
    def __init__(self, input_size=2048 * 11, output_size=460):
        super(RegressionModel, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),                    # Input flattening layer
            nn.Linear(input_size, 2048),
            nn.BatchNorm1d(2048),     # First hidden layer
            nn.ReLU(),               
            nn.Linear(2048, 1024),           # Third hidden layer
            nn.BatchNorm1d(1024),     # First hidden layer
            nn.ReLU(),
            nn.Linear(1024, output_size)      # Output layer
        )

    def forward(self, x):
        return self.model(x)


class CNN_regression_reduced_model(nn.Module):
    def __init__(self, input_height=16, input_width=1024, output_size=460):
        super(CNN_regression_reduced_model, self).__init__()

        # Set kernel sizes
        kernel_size_1 = (8, 1)
        kernel_size_2 = (1, 100)
        kernel_size_3 = (3, 3)
        pool_size_1 = (1, 2)  # Max pooling size
        pool_size_2 = (1, 10)
        pool_size_3 = (2, 2)

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=kernel_size_1)
        self.pool1 = nn.MaxPool2d(pool_size_1)
        self.bn1 = nn.BatchNorm2d(128)

        self.conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=kernel_size_2)
        self.pool2 = nn.MaxPool2d(pool_size_2)
        self.bn2 = nn.BatchNorm2d(128)

        # self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=kernel_size_3)
        # self.pool3 = nn.MaxPool2d(pool_size_3)
        # self.bn3 = nn.BatchNorm2d(64)
        # self.conv3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=kernel_size_3)

        # Compute output dimensions after convolutions and pooling
        conv1_out_height, conv1_out_width = self.compute_conv_output_size(input_height, input_width, kernel_size_1)
        # print(f"conv1 output size: ({conv1_out_height}, {conv1_out_width})")
        
        pool1_out_height, pool1_out_width = self.compute_pool_output_size(conv1_out_height, conv1_out_width, pool_size_1)
        # print(f"pool1 output size: ({pool1_out_height}, {pool1_out_width})")

        conv2_out_height, conv2_out_width = self.compute_conv_output_size(pool1_out_height, pool1_out_width, kernel_size_2)
        # print(f"conv2 output size: ({conv2_out_height}, {conv2_out_width})")
        
        pool2_out_height, pool2_out_width = self.compute_pool_output_size(conv2_out_height, conv2_out_width, pool_size_2)
        # print(f"pool2 output size: ({pool2_out_height}, {pool2_out_width})")

        # conv3_out_height, conv3_out_width = self.compute_conv_output_size(pool2_out_height, pool2_out_width, kernel_size_3)
        # print(f"conv3 output size: ({conv3_out_height}, {conv3_out_width})")

        # pool3_out_height, pool3_out_width = self.compute_pool_output_size(conv3_out_height, conv3_out_width, pool_size_3)

        # Dynamically calculate flattened size based on actual output dimensions after conv3
        self.flattened_size = 128 * pool2_out_height * pool2_out_width  # Update flattened size
        # print(f"Flattened size: {self.flattened_size}")

        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, 1024)  # Use calculated flattened_size
        self.bn_fc = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, output_size)

    def compute_conv_output_size(self, input_height, input_width, kernel_size, stride=1, padding=0):
        # Formula to calculate output size after convolution or pooling
        output_height = (input_height - kernel_size[0] + 2 * padding) // stride + 1
        output_width = (input_width - kernel_size[1] + 2 * padding) // stride + 1
        return output_height, output_width

    def compute_pool_output_size(self, input_height, input_width, kernel_size, padding=0):
        # Formula to calculate output size after convolution or pooling with stride=2
        output_height = (input_height - kernel_size[0] + 2 * padding) // kernel_size[0] + 1
        output_width = (input_width - kernel_size[1] + 2 * padding) // kernel_size[1] + 1
        
        return output_height, output_width

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        # print(f"Shape after conv1: {x.shape}")  # Debugging shape
        
        x = self.pool1(x)
        # print(f"Shape after pool1: {x.shape}")  # Debugging shape

        x = F.relu(self.bn2(self.conv2(x)))
        # print(f"Shape after conv2: {x.shape}")  # Debugging shape
        
        x = self.pool2(x)
        # print(f"Shape after pool2: {x.shape}")  # Debugging shape

        # x = F.relu(self.conv3(x))
        # print(f"Shape after conv3: {x.shape}")  # Debugging shape

        # x = F.relu(self.bn3(self.conv3(x)))
        # print(f"Shape after conv2: {x.shape}")  # Debugging shape
        
        # x = self.pool3(x)
        # print(f"Shape after pool2: {x.shape}")  # Debugging shape


        x = x.view(x.size(0), -1)  # Flatten the tensor
        # print(f"Shape after flatten: {x.shape}")  # Debugging shape

        x = F.relu(self.bn_fc(self.fc1(x)))
        # print(f"Shape after fc1: {x.shape}")  # Debugging shape
        
        x = self.fc2(x)
        return x


############# New Model


class CNN_regression_model_three_fc(nn.Module):
    def __init__(self, input_height=16, input_width=1024, output_size=460):
        super(CNN_regression_model_three_fc, self).__init__()

        # Set kernel sizes
        kernel_size_1 = (8, 1)
        kernel_size_2 = (1, 100)
        kernel_size_3 = (3, 3)
        pool_size_1 = (1, 2)  # Max pooling size
        pool_size_2 = (1, 10)
        pool_size_3 = (2, 2)

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=kernel_size_1)
        self.pool1 = nn.MaxPool2d(pool_size_1)
        self.bn1 = nn.BatchNorm2d(128)

        self.conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=kernel_size_2)
        self.pool2 = nn.MaxPool2d(pool_size_2)
        self.bn2 = nn.BatchNorm2d(128)

        # self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=kernel_size_3)
        # self.pool3 = nn.MaxPool2d(pool_size_3)
        # self.bn3 = nn.BatchNorm2d(64)
        # self.conv3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=kernel_size_3)

        # Compute output dimensions after convolutions and pooling
        conv1_out_height, conv1_out_width = self.compute_conv_output_size(input_height, input_width, kernel_size_1)
        # print(f"conv1 output size: ({conv1_out_height}, {conv1_out_width})")
        
        pool1_out_height, pool1_out_width = self.compute_pool_output_size(conv1_out_height, conv1_out_width, pool_size_1)
        # print(f"pool1 output size: ({pool1_out_height}, {pool1_out_width})")

        conv2_out_height, conv2_out_width = self.compute_conv_output_size(pool1_out_height, pool1_out_width, kernel_size_2)
        # print(f"conv2 output size: ({conv2_out_height}, {conv2_out_width})")
        
        pool2_out_height, pool2_out_width = self.compute_pool_output_size(conv2_out_height, conv2_out_width, pool_size_2)
        # print(f"pool2 output size: ({pool2_out_height}, {pool2_out_width})")

        # conv3_out_height, conv3_out_width = self.compute_conv_output_size(pool2_out_height, pool2_out_width, kernel_size_3)
        # print(f"conv3 output size: ({conv3_out_height}, {conv3_out_width})")

        # pool3_out_height, pool3_out_width = self.compute_pool_output_size(conv3_out_height, conv3_out_width, pool_size_3)

        # Dynamically calculate flattened size based on actual output dimensions after conv3
        self.flattened_size = 128 * pool2_out_height * pool2_out_width  # Update flattened size
        # print(f"Flattened size: {self.flattened_size}")

        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, 4096)  # Use calculated flattened_size
        self.bn_fc = nn.BatchNorm1d(4096)
        self.fc2 = nn.Linear(4096, 1024)
        self.bn_fc_1 = nn.BatchNorm1d(1024)
        self.fc3 = nn.Linear(1024, output_size)

    def compute_conv_output_size(self, input_height, input_width, kernel_size, stride=1, padding=0):
        # Formula to calculate output size after convolution or pooling
        output_height = (input_height - kernel_size[0] + 2 * padding) // stride + 1
        output_width = (input_width - kernel_size[1] + 2 * padding) // stride + 1
        return output_height, output_width

    def compute_pool_output_size(self, input_height, input_width, kernel_size, padding=0):
        # Formula to calculate output size after convolution or pooling with stride=2
        output_height = (input_height - kernel_size[0] + 2 * padding) // kernel_size[0] + 1
        output_width = (input_width - kernel_size[1] + 2 * padding) // kernel_size[1] + 1
        
        return output_height, output_width

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        # print(f"Shape after conv1: {x.shape}")  # Debugging shape
        
        x = self.pool1(x)
        # print(f"Shape after pool1: {x.shape}")  # Debugging shape

        x = F.relu(self.bn2(self.conv2(x)))
        # print(f"Shape after conv2: {x.shape}")  # Debugging shape
        
        x = self.pool2(x)
        # print(f"Shape after pool2: {x.shape}")  # Debugging shape

        # x = F.relu(self.conv3(x))
        # print(f"Shape after conv3: {x.shape}")  # Debugging shape

        # x = F.relu(self.bn3(self.conv3(x)))
        # print(f"Shape after conv2: {x.shape}")  # Debugging shape
        
        # x = self.pool3(x)
        # print(f"Shape after pool2: {x.shape}")  # Debugging shape


        x = x.view(x.size(0), -1)  # Flatten the tensor
        # print(f"Shape after flatten: {x.shape}")  # Debugging shape

        x = F.relu(self.bn_fc(self.fc1(x)))
        # print(f"Shape after fc1: {x.shape}")  # Debugging shape
        x = F.relu(self.bn_fc_1(self.fc2(x)))
        
        x = self.fc3(x)
        return x
