import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # Convolutional Layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        # Fully Connected Layers
        self.fc1 = nn.Linear(64 * 15 * 15, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 2)

        # Activation and Pooling Layers
        self.activation = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Dropout Layers
        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.5)

    def forward(self, x):
        # Convolutional Layer Block 1
        x = self.pool(self.activation(self.conv1(x)))
        # Convolutional Layer Block 2
        x = self.pool(self.activation(self.conv2(x)))
        # Convolutional Layer Block 3
        x = self.pool(self.activation(self.conv3(x)))
        # Convolutional Layer Block 4
        x = self.activation(self.conv4(x))

        # Flatten
        x = x.view(-1, 64 * 15 * 15)

        # Fully Connected Layers
        x = self.dropout1(self.activation(self.fc1(x)))
        x = self.dropout2(self.activation(self.fc2(x)))
        x = self.fc3(x)

        return F.softmax(x, dim=1)
    
    def count_parameters(self):
        total_params = 0
        for name, param in self.named_parameters():
            num_params = param.numel()
            total_params += num_params
            print(f"{name}: {num_params} parameters")
        print(f"Total number of parameters in the model: {total_params}")
        return total_params
