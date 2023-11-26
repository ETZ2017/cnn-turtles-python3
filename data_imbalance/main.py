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
import torchvision.transforms as transforms
from torchvision import datasets, models, transforms
import random
import os
from tqdm import tqdm
from tqdm import trange
from torchviz import make_dot

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score # Add metrics
from dataset import TurtleDataset
from model import CNN
from parser import parse_arguments
from collections import Counter

from torchinfo import summary
from pathlib import Path

params = parse_arguments()

lr=params.lr
is_test=params.is_test
epochs=params.epochs
batch_size = params.batch_size
label_smoothing = params.label_smoothing
l2_lambda = params.l2_lambda
enable_aug_rhf = params.enable_aug_rhf
enable_aug_rvf = params.enable_aug_rvf
enable_aug_rr = params.enable_aug_rr
enable_ins_weights = params.enable_ins_weights
enable_root_weights = params.enable_root_weights
enable_label_smoothing = params.enable_label_smoothing
path=params.output_folder + "_e" + str(params.epochs) + "_l" + str(params.lr) + "_b" + str(params.batch_size)  + "_l2-" + str(params.l2_lambda) + "_s" + str(params.label_smoothing)  + "_t" + str(int(time.time())) + "_rhf" + str(params.enable_aug_rhf) + "_rvf" + str(params.enable_aug_rvf) + "_rr" + str(params.enable_aug_rr) + "_ins" + str(params.enable_ins_weights) + "_root" + str(params.enable_root_weights) + "_ls" + str(params.enable_label_smoothing)
device = 'cuda'

Path(path).mkdir(exist_ok=True)

with open(path + "/" + "meta.txt", "w+") as file:
    file.write(str(params))

def START_seed():
    seed = 9
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
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

# 3. Define Dataset and Dataloaders
# Assuming your dataset is in a folder structure suitable for ImageFolder
# Replace 'path_to_train' and 'path_to_val' with your dataset paths

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # This is the mean and std for the pretrained ResNet50 model
])

# 1. Data augmentation

transforms_list = [
   transforms.Resize(224),
]

# This label augmentation techniques are activated or deactivated with the use of the parameters enable_aug_rhf, enable_aug_rvf and enable_aug_rr
if enable_aug_rhf:
    transforms_list.append(transforms.RandomHorizontalFlip(0.5))
    
if enable_aug_rvf:
    transforms_list.append(transforms.RandomVerticalFlip(0.5))

if enable_aug_rr:
    transforms_list.append(transforms.RandomRotation((5, 85)))

transforms_list.append(transforms.ToTensor())
transforms_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

augmentation_transform = transforms.Compose(transforms_list)

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
sample_weights = [class_weights[int(label)] for label in train_labels] # More probability of sampling 1s

def get_weights_inverse_num_of_samples (no_of_classes, samples_per_cls, power = 1): # 2. Compute weights for the classes

    weights_for_samples = 1.0 / np.array(np.power(samples_per_cls, power))
    weights_for_samples = weights_for_samples / np.sum(weights_for_samples) * no_of_classes
    
    return weights_for_samples

def get_weights_square_root_inverse_num_of_samples (no_of_classes, samples_per_cls, power = 1/2): # Square inverse weights
    weights_for_samples_root= 1.0 / np.array(np.power(samples_per_cls, power))
    weights_for_samples_root = weights_for_samples_root / np.sum(weights_for_samples_root) * no_of_classes
    
    return weights_for_samples_root

no_of_classes = 2
samples_per_cls = [class_counts['0'] , class_counts['1']]

ins_class_weights = torch.tensor(get_weights_inverse_num_of_samples(no_of_classes, samples_per_cls)).float().to(device)

root_class_weights = torch.tensor(get_weights_square_root_inverse_num_of_samples(no_of_classes, samples_per_cls)).float().to(device)

print(ins_class_weights, root_class_weights)

sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=False)

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

def train_epoch(model, train_dataloader, criterion, optimizer):

    model.train()

    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with tqdm(train_dataloader, unit="batch") as tepoch:
            
        for batch_idx, (inputs, labels) in enumerate(tepoch):
            
            regularization_loss = 0.0
        
            # send to device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            # # Add L2 regularization to the loss function
            # regularization_loss = 0

            for param in model.parameters():
                regularization_loss += torch.norm(param)

            # Total loss with L2 regularization
            loss = criterion(outputs, labels) + 0.5 * l2_lambda * regularization_loss

            # loss = criterion(outputs, labels)

            # outputs = F.softmax(outputs,1)

            # loss = criterion(outputs[torch.arange(outputs.shape[0]), labels], labels)

            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs, 1)
            running_loss += loss.item()
            correct_predictions += (predicted == labels.data).sum().item()
            total_samples += labels.size(0)
            tepoch.set_postfix(loss=running_loss/(batch_idx+1), lr=optimizer.param_groups[0]['lr'])

    log = '{} train loss: {:.4f} accuracy: {:.4f}\n'.format(epoch, running_loss/(batch_idx+1), 100.*correct_predictions/total_samples)

    accuracy = correct_predictions / total_samples
    average_loss = running_loss / len(train_dataloader)

    with open(path + "/log.txt", 'a') as file:
                file.write(log)

    return average_loss, accuracy                

best_f1 = 0.0

def evaluate(model, dataloader, criterion):

    model.eval()

    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    all_predictions = []
    all_targets = []

    with tqdm(dataloader, unit="batch") as tepoch:

        with torch.no_grad():

            for batch_idx, (inputs, labels) in enumerate(tepoch):

                regularization_loss = 0.0
        
                # send to device
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)

                for param in model.parameters():
                    regularization_loss += torch.norm(param)
                    
                # Total loss with L2 regularization
                loss = criterion(outputs, labels) + 0.5 * l2_lambda * regularization_loss

                # loss = criterion(outputs, labels)

                # outputs = F.softmax(outputs,1)

                # loss = criterion(outputs[torch.arange(outputs.shape[0]), labels], labels)

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct_predictions += (predicted == labels.data).sum().item()
                total_samples += labels.size(0)

                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())

    accuracy = correct_predictions / total_samples
    average_loss = running_loss / len(dataloader)

    log = ' val loss: {:.4f} accuracy: {:.4f} best_accuracy: {:.4f}\n'.format(running_loss/(batch_idx+1), 100.*correct_predictions/total_samples, best_f1)
    
    print(log)
    
    with open(path + "/log.txt", 'a') as file:
            file.write(log)

    return average_loss, accuracy, all_predictions, all_targets                             

def test(model, test_dataloader, criterion, best_model_path):

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
            
                regularization_loss = 0.0

                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)

                # loss = criterion(outputs, labels)

                for param in model.parameters():
                    regularization_loss += torch.norm(param)
                    
                # Total loss with L2 regularization
                loss = criterion(outputs, labels) + 0.5 * l2_lambda * regularization_loss

                # outputs = F.softmax(outputs,1)

                # loss = criterion(outputs[torch.arange(outputs.shape[0]), labels], labels)

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

# Focal loss, as indicated by Lin et al. in https://arxiv.org/pdf/1708.02002.pdf
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets.float(), reduction='none')
        pt = torch.exp(-BCE_loss)  # prevents nans when probability 0
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

if __name__ == "__main__":

    START_seed()
    #train_loader, val_loader, test_loader = load_dataset()

    run = wandb.init(
        entity="turtle_detection",
        config={
            "epochs": params.epochs,
            "batch_size": params.batch_size,
            "lr": params.lr,
        },
        name="turtles"
    )

    # 1. Load Pre-trained ResNet50 Model
    model = models.resnet50(pretrained=True)

    # 2. Modify the Final Layer
    num_features = model.fc.in_features
    num_classes = 2  
    model.fc = nn.Linear(num_features, num_classes)

    # if enable_ins_weights:
    #     if enable_label_smoothing:
    #         criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing, weight=ins_class_weights) #Trying different types of weights and label smoothing
    #     else: 
    #         criterion = nn.CrossEntropyLoss(weight=ins_class_weights) #Trying different types of weights without label smoothing
    # elif enable_root_weights:
    #     if enable_label_smoothing:
    #         criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing, weight=root_class_weights) #Trying different types of weights and label smoothing
    #     else: 
    #         criterion = nn.CrossEntropyLoss(weight=root_class_weights) #Trying different types of weights without label smoothing
    # else:
    #     if enable_label_smoothing:
    #         criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing) #Trying different types of weights. Default weights and label smoothing
    #     else: 
    #         criterion = nn.CrossEntropyLoss() #Trying different types of weights. Default weights and no label smoothing

    criterion = nn.CrossEntropyLoss(weight=ins_class_weights)

    # criterion = nn.CrossEntropyLoss(weight = root_class_weights)

    # criterion = FocalLoss()

    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=l2_lambda, nesterov=True) # Weight decay works as L2 regularization
    # optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2_lambda) # Weight decay works as L2 regularization
    # optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2_lambda)
    optimizer = optim.Adam(model.parameters(), lr=lr)


    pytorch_total_params = sum(p.numel() for p in  model.parameters())
    print('Number of parameters: {0}'.format(pytorch_total_params))

    model.to(device)
    start = time.time()
    
    if is_test:

        for epoch in range(0, epochs):

            train_loss, train_accuracy = train_epoch(model, train_loader, criterion, optimizer)
            print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")

            val_loss, val_accuracy, all_predictions, all_targets  = evaluate(model, val_loader, criterion)

            # Calculate metrics using sklearn.metrics

            precision = precision_score(all_targets, all_predictions)
            recall = recall_score(all_targets, all_predictions)
            f1 = f1_score(all_targets, all_predictions)
            roc_auc = roc_auc_score(all_targets, all_predictions)

            if f1 > best_f1:

                print("Saving the best model...")
                torch.save(model.state_dict(), path + '/best_model.pth')

                best_f1 = f1

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

    else:
        
        best_model_path = 'models/best_model.pth' 
    
    test_loss, test_accuracy, test_all_predictions, test_all_targets = test(model, test_loader, criterion, best_model_path)
    
    test_precision = precision_score(test_all_targets, test_all_predictions)
    test_recall = recall_score(test_all_targets, test_all_predictions)
    test_f1 = f1_score(test_all_targets, test_all_predictions)
    test_roc_auc = roc_auc_score(test_all_targets, test_all_predictions)

    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}")
    print(f"Test F1: {test_f1:.4f}, Test ROC: {test_roc_auc:.4f}")

    wandb.log({
        'test/acccuracy': test_accuracy,
        'test/loss': test_loss,
        'test/precision': test_precision,
        'test/recall': test_recall,# 2. Modify the Final Layer
        'test/roc_auc': test_roc_auc,
    })

    end = time.time()
    Total_time=end-start
    print('Total training and inference time is: {0}'.format(Total_time))

    run.finish()