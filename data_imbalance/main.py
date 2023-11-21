#importing libraries

import os
import numpy as np
import torch
import time
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import random
import cv2
import glob
from PIL import Image
import ntpath
import os
from tqdm import tqdm
from tqdm import trange

from dataset import TurtleDataset
from model import CNN
from parser import parse_arguments
from matplotlib import pyplot as plt

from torchinfo import summary
from pathlib import Path

params = parse_arguments()

lr=params.lr
epochs=params.epochs
optimizer=params.opt
batch_size = 64
path=params.output_folder + "_e" + str(params.epochs) + "_l" + str(params.lr) + "_" + str(params.opt) + "_" + str(params.batch_size)
device = 'cuda'

Path(path).mkdir(exist_ok=True)

with open(path + "/" + "meta.txt", "w+") as file:
    file.write(str(params))

def START_seed():
    seed = 9
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# Create a custom dataset for training
train_dataset = TurtleDataset(transform=True, train=True)
# Create a custom dataset for testing
test_dataset = TurtleDataset(transform=True, train=False)

print("Test set length: ", len(test_dataset))    

## split into train, val, test 
print("Train set length: ", len(train_dataset))     
val_size = int(0.1 * len(train_dataset))
print("Validation set length: ", val_size)
train_size = len(train_dataset) - val_size
print("Train set length after split for validation", train_size)
train, val = torch.utils.data.random_split(train_dataset, [train_size, val_size])  

print(train, val)

train_loader = torch.utils.data.DataLoader(
    train, batch_size=batch_size, shuffle=True, num_workers=4
)
val_loader = torch.utils.data.DataLoader(
    val, batch_size=batch_size, shuffle=False, num_workers=4
)    
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=4
)

def train_epoch(model, train_dataloader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    with tqdm(train_dataloader, unit="batch") as tepoch:
            for batch_idx, (inputs, labels) in enumerate(tepoch):
                # send to device
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_samples += labels.size(0)
                tepoch.set_postfix(loss=running_loss/(batch_idx+1), lr=optimizer.param_groups[0]['lr'])

    log = '{} train loss: {:.4f} accuracy: {:.4f}\n'.format(epoch, running_loss/(batch_idx+1), 100.*correct_predictions/total_samples)
    accuracy = correct_predictions / total_samples
    average_loss = running_loss / len(train_dataloader)
    with open(path + "/log.txt", 'a') as file:
                file.write(log)

    return average_loss, accuracy                

best_accuracy = 0.0
def evaluate(model, dataloader, criterion):
    global best_accuracy
    model.eval()

    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with tqdm(dataloader, unit="batch") as tepoch:
        with torch.no_grad():
            for batch_idx, (inputs, labels) in dataloader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_samples += labels.size(0)

    accuracy = correct_predictions / total_samples
    average_loss = running_loss / len(dataloader)

    log = ' val loss: {:.4f} accuracy: {:.4f} best_accuracy: {:.4f}\n'.format(running_loss/(batch_idx+1), 100.*correct_predictions/total_samples, best_accuracy)
    print(log)
    with open(path + "/log.txt", 'a') as file:
            file.write(log)        
    if (100.*correct_predictions/total_samples) > best_accuracy:
        print("Saving the best model...")
        best_accuracy = (100.*correct_predictions/total_samples)
        torch.save(model.state_dict(), path + '/best_model.pth')

    return average_loss, accuracy                                    

def test(model, test_dataloader, criterion,  best_model_path):
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    test_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    with tqdm(test_dataloader, unit="batch") as tepoch:
        with torch.no_grad():
             for batch_idx, (inputs, labels) in test_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                test_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_samples += labels.size(0)

    accuracy = correct_predictions / total_samples
    average_loss = test_loss / len(test_dataloader)
   
    log = 'Test loss: {:.4f} Accuracy: {:.2f}%'.format(test_loss/(batch_idx+1), accuracy)
    print(log)
    with open(path + "/log.txt", 'a') as file:
        file.write(log)  

    return average_loss, accuracy              

if __name__ == "__main__":

    START_seed()
    #train_loader, val_loader, test_loader = load_dataset()

    model = CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0, nesterov=True)

    pytorch_total_params = sum(p.numel() for p in  model.parameters())
    print('Number of parameters: {0}'.format(pytorch_total_params))

    model.to(device)
    start = time.time()

    for epoch in range(0, epochs):
        train_loss, train_accuracy = train_epoch(model, train_loader, criterion, optimizer)
        print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")

    val_loss, val_accuracy = evaluate(model, val_loader, criterion)
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    best_model_path = path + '/best_model.pth' 
    test_loss, test_accuracy = test(model, test_loader, criterion, best_model_path)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    end = time.time()
    Total_time=end-start
    print('Total training and inference time is: {0}'.format(Total_time))