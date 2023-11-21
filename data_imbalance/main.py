#importing libraries

import os
import numpy as np
import torch
import time
import wandb
import torch.utils.data
from torch.utils.data import WeightedRandomSampler
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

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from dataset import TurtleDataset
from model import CNN
from parser import parse_arguments
from collections import Counter

from torchinfo import summary
from pathlib import Path

params = parse_arguments()

lr=params.lr
epochs=params.epochs
batch_size = params.batch_size
label_smoothing = params.label_smoothing
l2_lambda = params.l2_lambda
path=params.output_folder + "_e" + str(params.epochs) + "_l" + str(params.lr) + "_b" + str(params.batch_size)  + "_l2" + str(params.l2_lambda) + "_s" + str(params.label_smoothing)  + "_t" + str(int(time.time())) 
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

transform = transforms.Compose([
    transforms.Resize((120, 120)),  # Adjust the size as needed
    transforms.ToTensor(),
])

augmentation_transform = transforms.Compose([
    # Add your desired data augmentation transformations here
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.5),
    transforms.RandomRotation((5, 85)),
    transforms.Resize((120, 120)),  # Adjust the size as needed
    transforms.ToTensor(),
])

multiple_transforms = [transform, augmentation_transform]

# Create a custom dataset for testing
train_dataset = TurtleDataset(transform=multiple_transforms, train=True)
test_dataset = TurtleDataset(transform=transform, train=False)

print("Test set length: ", len(test_dataset))

## split into train, val, test 
print("Train set length: ", len(train_dataset))     
val_size = int(0.3 * len(train_dataset))
print("Validation set length: ", val_size)
train_size = len(train_dataset) - val_size
print("Train set length after split for validation", train_size)
train, val = torch.utils.data.random_split(train_dataset, [train_size, val_size])  

# Calculate class weights to balance the dataset

train_labels = train_dataset.get_labels_from_indices(train.indices)

class_counts = Counter(train_labels)
class_weights = [1.0 / class_counts[label] for label in train_labels]
sample_weights = [class_weights[int(label)] for label in train_labels]

def get_weights_inverse_num_of_samples (no_of_classes, samples_per_cls, power = 1):

    weights_for_samples = 1.0 / np.array(np.power(samples_per_cls, power))
    weights_for_samples = weights_for_samples / np.sum(weights_for_samples) * no_of_classes
    
    return weights_for_samples

no_of_classes = 2
samples_per_cls = [class_counts['0'] , class_counts['1']]

ins_class_weights = torch.tensor(get_weights_inverse_num_of_samples(no_of_classes, samples_per_cls)).float().to(device)

sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = torch.utils.data.DataLoader(
    train, batch_size=batch_size, num_workers=8, sampler=sampler
)
val_loader = torch.utils.data.DataLoader(
    val, batch_size=batch_size, shuffle=False, num_workers=8,
)    
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=8,
)

def calculate_class_distribution(targets):
    class_counts = {}
    for label in targets:
        if label not in class_counts:
            class_counts[label] = 0
        class_counts[label] += 1
    return class_counts

def train_epoch(model, train_dataloader, criterion, optimizer, l2_lambda = 0.01):
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

                # Add L2 regularization to the loss function
                regularization_loss = 0

                for param in model.parameters():
                    regularization_loss += torch.norm(param, p=2)  # L2 norm

                # Total loss with L2 regularization
                loss = criterion(outputs, labels) + 0.5 * l2_lambda * regularization_loss

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
    
    all_predictions = []
    all_targets = []

    with tqdm(dataloader, unit="batch") as tepoch:
        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(tepoch):

                # send to device
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_samples += labels.size(0)

                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())

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

    return average_loss, accuracy, all_predictions, all_targets                             

def test(model, test_dataloader, criterion,  best_model_path):
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    test_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    all_predictions = []
    all_targets = []

    with tqdm(test_dataloader, unit="batch") as tepoch:
        with torch.no_grad():
             for batch_idx, (inputs, labels) in enumerate(tepoch):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                test_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_samples += labels.size(0)

                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())

    accuracy = correct_predictions / total_samples
    average_loss = test_loss / len(test_dataloader)
   
    log = 'Test loss: {:.4f} Accuracy: {:.2f}%'.format(test_loss/(batch_idx+1), accuracy)
    print(log)
    with open(path + "/log.txt", 'a') as file:
        file.write(log)  

    return average_loss, accuracy, all_predictions, all_targets             

if __name__ == "__main__":

    START_seed()
    #train_loader, val_loader, test_loader = load_dataset()

    run = wandb.init(
        entity="evhenia-k-you",
        config={
            "epochs": params.epochs,
            "batch_size": params.batch_size,
            "lr": params.lr,
            "opt": params.opt,
        },
        name="turtles"
    )

    model = CNN()

    criterion = nn.CrossEntropyLoss(label_smoothing=0.2, weight=ins_class_weights)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0, nesterov=True)

    pytorch_total_params = sum(p.numel() for p in  model.parameters())
    print('Number of parameters: {0}'.format(pytorch_total_params))

    model.to(device)
    start = time.time()

    for epoch in range(0, epochs):

        train_loss, train_accuracy = train_epoch(model, train_loader, criterion, optimizer)
        print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")

        val_loss, val_accuracy, all_predictions, all_targets  = evaluate(model, val_loader, criterion)

        # Calculate metrics using sklearn.metrics

        precision = precision_score(all_targets, all_predictions)
        recall = recall_score(all_targets, all_predictions)
        f1 = f1_score(all_targets, all_predictions)
        roc_auc = roc_auc_score(all_targets, all_predictions)

        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
        print(f"Validation Precision: {precision:.4f}, Validation Recall: {recall:.4f}")
        print(f"Validation F1: {f1:.4f}, Validation ROC: {roc_auc:.4f}")
        
        wandb.log({
            'train/acccuracy': train_accuracy,
            'train/loss': train_loss,
            'eval/acccuracy': val_accuracy,
            'eval/loss': val_loss,
            'eval/precision': precision,
            'eval/recall': recall,
            'eval/f1': f1,
            'eval/roc_auc': roc_auc,
        })

    best_model_path = path + '/best_model.pth' 
    test_loss, test_accuracy, test_all_predictions, test_all_targets = test(model, test_loader, criterion, best_model_path)
    
    test_precision = precision_score(all_targets, all_predictions)
    test_recall = recall_score(all_targets, all_predictions)
    test_f1 = f1_score(all_targets, all_predictions)
    test_roc_auc = roc_auc_score(all_targets, all_predictions)

    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}")
    print(f"Test F1: {test_f1:.4f}, Test ROC: {test_roc_auc:.4f}")

    wandb.log({
        'test/acccuracy': test_accuracy,
        'test/loss': test_loss,
        'test/precision': test_precision,
        'test/recall': test_recall,
        'test/f1': test_f1,
        'test/roc_auc': test_roc_auc,
    })

    end = time.time()
    Total_time=end-start
    print('Total training and inference time is: {0}'.format(Total_time))

    run.finish()