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
from torchvision import transforms
import random
import os
from tqdm import tqdm
from tqdm import trange

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from dataset import TurtleDataset
from dilated_model import DilatedCNNObjectDetection
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
path=params.output_folder + "_e" + str(params.epochs) + "_l" + str(params.lr) + "_b" + str(params.batch_size)  + "_l2-" + str(params.l2_lambda) + "_s" + str(params.label_smoothing)  + "_t" + str(int(time.time())) 
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
    # transforms.Normalize()
])

augmentation_transform = transforms.Compose([
    # Add your desired data augmentation transformations here
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.5),
    transforms.RandomRotation((5, 85)),
    transforms.Resize((120, 120)),  # Adjust the size as needed
    transforms.ToTensor(),
    # transforms.Normalize()
])

multiple_transforms = [transform, augmentation_transform]

# Create a custom dataset for testing
train_dataset = TurtleDataset(transform=multiple_transforms, train=True)
test_dataset = TurtleDataset(transform=transform, train=False)

print("Test set length: ", len(test_dataset))

## split into train, val, test 
print("Train set length: ", len(train_dataset))     
val_size = int(0.15 * len(train_dataset))
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

def calculate_iou(pred_boxes, top, left, size = 25):

    # Extract coordinates and dimensions
    pred_x1, pred_y1 = pred_boxes.unbind(dim=2)
    target_x1, target_y1 = left, top

    # Calculate intersection coordinates
    intersection_x1 = torch.maximum(pred_x1, target_x1)
    intersection_y1 = torch.maximum(pred_y1, target_y1)
    
    # Calculate intersection area
    intersection_width = torch.maximum(torch.zeros_like(intersection_x1), torch.minimum(pred_x1 + size, target_x1 + size) - intersection_x1)
    intersection_height = torch.maximum(torch.zeros_like(intersection_x1), torch.minimum(pred_y1 + size, target_y1 + size) - intersection_y1)
    intersection_area = intersection_width * intersection_height

    # Calculate union area
    pred_area = size * size
    target_area = size * size
    union_area = pred_area + target_area - intersection_area

    # Calculate IoU
    iou = intersection_area / torch.clamp(union_area, min=1e-5)  # Avoid division by zero

    return iou

def train_epoch(model, train_dataloader, optimizer):
    model.train()
    running_loss = 0.0
    total_cls_correct = 0
    total_iou = 0
    total_samples = 0
    with tqdm(train_dataloader, unit="batch") as tepoch:
            for batch_idx, (inputs, targets_cls, top, left) in enumerate(tepoch):
                # send to device
                optimizer.zero_grad()
                inputs, targets_cls, top, left = inputs.to(device), targets_cls.to(device), top.to(device), left.to(device)

                pred_class, pred_box = model(inputs)

                # Calculate classification accuracy
                cls_preds = torch.argmax(pred_class, dim=1)
                cls_correct = torch.sum(cls_preds == targets_cls)
                total_cls_correct += cls_correct.item()

                # Calculate bounding box accuracy (considering top-left point)
                iou = calculate_iou(pred_box, top, left)
                total_iou += iou.sum().item()

                real_box = torch.stack([top, left])

                loss = model.calculate_loss(pred_class, targets_cls, pred_box, real_box)
                running_loss += loss.item()

                loss.backward()
                optimizer.step()

                tepoch.set_postfix(loss=running_loss/(batch_idx+1), lr=optimizer.param_groups[0]['lr'])

                total_samples += inputs.size(0)

    average_loss = running_loss / len(train_dataloader)
    avg_cls_accuracy = total_cls_correct / total_samples
    avg_bbox_accuracy = total_iou / total_samples
    
    log = '{} train loss: {:.4f} average classification accuracy: {:.4f} average bbox accuracy: {:.4f}\n'.format(epoch, running_loss/(batch_idx+1), 100.*avg_cls_accuracy, avg_bbox_accuracy)
    with open(path + "/log.txt", 'a') as file:
                file.write(log)

    return average_loss, avg_cls_accuracy, avg_bbox_accuracy                

best_accuracy = 0.0
def evaluate(model, dataloader):
    global best_accuracy
    model.eval()

    running_loss = 0.0
    total_cls_correct = 0
    total_iou = 0
    total_samples = 0
    
    all_cls_predictions = []
    all_cls_targets = []

    with tqdm(dataloader, unit="batch") as tepoch:
        with torch.no_grad():
            for batch_idx, (inputs, targets_cls, top, left) in enumerate(tepoch):

                # send to device
                inputs, targets_cls, top, left = inputs.to(device), targets_cls.to(device), top.to(device), left.to(device)

                pred_class, pred_box = model(inputs)

                # Calculate classification accuracy
                cls_preds = torch.argmax(pred_class, dim=1)
                cls_correct = torch.sum(cls_preds == targets_cls)
                total_cls_correct += cls_correct.item()

                # Calculate bounding box accuracy (considering top-left point)
                iou = calculate_iou(pred_box, top, left)
                total_iou += iou.sum().item()

                real_box = torch.stack([top, left])

                loss = model.calculate_loss(pred_class, targets_cls, pred_box, real_box)
                running_loss += loss.item()

                total_samples += inputs.size(0)

                all_cls_predictions.extend(cls_preds.cpu().numpy())
                all_cls_targets.extend(targets_cls.cpu().numpy())

    average_loss = running_loss / len(dataloader)
    avg_cls_accuracy = total_cls_correct / total_samples
    avg_bbox_accuracy = total_iou / total_samples

    log = '{} train loss: {:.4f} average classification accuracy: {:.4f} average bbox accuracy: {:.4f}\n'.format(epoch, running_loss/(batch_idx+1), 100.*avg_cls_accuracy, avg_bbox_accuracy)
    with open(path + "/log.txt", 'a') as file:
                file.write(log)


    if (100.*total_cls_correct/total_samples) > best_accuracy:
        print("Saving the best model...")
        best_accuracy = (100.*total_cls_correct/total_samples)
        torch.save(model.state_dict(), path + '/best_model.pth')

    return average_loss, avg_cls_accuracy, avg_bbox_accuracy, all_cls_predictions, all_cls_targets, total_iou                             

def test(model, test_dataloader,  best_model_path):
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    test_loss = 0.0
    total_cls_correct = 0
    total_iou = 0
    total_samples = 0
    
    all_cls_predictions = []
    all_cls_targets = []

    with tqdm(test_dataloader, unit="batch") as tepoch:
        with torch.no_grad():
              for batch_idx, (inputs, targets_cls, top, left) in enumerate(tepoch):

                # send to device
                inputs, targets_cls, top, left = inputs.to(device), targets_cls.to(device), top.to(device), left.to(device)

                pred_class, pred_box = model(inputs)

                # Calculate classification accuracy
                cls_preds = torch.argmax(pred_class, dim=1)
                cls_correct = torch.sum(cls_preds == targets_cls)
                total_cls_correct += cls_correct.item()

                # Calculate bounding box accuracy (considering top-left point)
                iou = calculate_iou(pred_box, top, left)
                total_iou += iou.sum().item()

                real_box = torch.stack([top, left])

                test_loss = model.calculate_loss(pred_class, targets_cls, pred_box, real_box)
                total_samples += inputs.size(0)

                all_cls_predictions.extend(cls_preds.cpu().numpy())
                all_cls_targets.extend(targets_cls.cpu().numpy())

    average_loss = test_loss / len(test_dataloader)
    avg_cls_accuracy = total_cls_correct / total_samples
    avg_bbox_accuracy = total_iou / total_samples
   
    log = 'Test loss: {:.4f} Class Accuracy: {:.2f}% Bounding box accuracy {:.2f}'.format(test_loss/(batch_idx+1), 100.*avg_cls_accuracy, avg_bbox_accuracy)
    with open(path + "/log.txt", 'a') as file:
        file.write(log)  

    return average_loss, avg_cls_accuracy, avg_bbox_accuracy, all_cls_predictions, all_cls_targets             

if __name__ == "__main__":

    START_seed()
    #train_loader, val_loader, test_loader = load_dataset()

    # run = wandb.init(
    #     entity="evhenia-k-you",
    #     config={
    #         "epochs": params.epochs,
    #         "batch_size": params.batch_size,
    #         "lr": params.lr,
    #         "opt": params.opt,
    #     },
    #     name="turtles"
    # )

    model = DilatedCNNObjectDetection()

    # criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing, weight=ins_class_weights)
    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=l2_lambda, nesterov=True)
    # optimizer = optim.Adam(model.parameters(), lr=lr, momentum=0.9, weight_decay=l2_lambda, nesterov=True)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=l2_lambda)

    pytorch_total_params = sum(p.numel() for p in  model.parameters())
    print('Number of parameters: {0}'.format(pytorch_total_params))

    model.to(device)
    start = time.time()

    for epoch in range(0, epochs):

        train_loss, train_cls_accuracy, train_bbox_accuracy  = train_epoch(model, train_loader, optimizer)
        print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}, Train Cls Accuracy: {train_cls_accuracy:.4f}, Train Bbox Accuracy: {train_bbox_accuracy:.4f}")

        val_loss, val_cls_accuracy, val_bbox_accuracy, all_cls_predictions, all_cls_targets, total_iou  = evaluate(model, val_loader)
        # Calculate metrics using sklearn.metrics

        precision = precision_score(all_cls_targets, all_cls_predictions)
        recall = recall_score(all_cls_targets, all_cls_predictions)
        f1 = f1_score(all_cls_targets, all_cls_predictions)
        roc_auc = roc_auc_score(all_cls_targets, all_cls_predictions)

        print(f"Validation Loss: {val_loss:.4f}, Validation Classification Accuracy: {val_cls_accuracy:.4f}, Validation Bbox Accuracy: {val_bbox_accuracy:.4f}")
        print(f"Validation Precision: {precision:.4f}, Validation Recall: {recall:.4f}")
        print(f"Validation F1: {f1:.4f}, Validation ROC: {roc_auc:.4f}")
        
        # wandb.log({
        #     'train/acccuracy': train_accuracy,
        #     'train/loss': train_loss,
        #     'eval/acccuracy': val_accuracy,
        #     'eval/loss': val_loss,
        #     'eval/precision': precision,
        #     'eval/recall': recall,
        #     'eval/f1': f1,
        #     'eval/roc_auc': roc_auc,
        # })

    best_model_path = path + '/best_model.pth' 
    test_loss, test_cls_accuracy, test_bbox_accuracy, test_cls_predictions, test_cls_targets = test(model, test_loader, best_model_path)
    
    test_precision = precision_score(test_cls_targets, test_cls_predictions)
    test_recall = recall_score(test_cls_targets, test_cls_predictions)
    test_f1 = f1_score(test_cls_targets, test_cls_predictions)
    test_roc_auc = roc_auc_score(test_cls_targets, test_cls_predictions)

    print(f"Test Loss: {test_loss:.4f}, Test Classification Accuracy: {test_cls_accuracy:.4f}, Test Bbox Accuracy: {test_bbox_accuracy:.4f}")
    print(f"Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}")
    print(f"Test F1: {test_f1:.4f}, Test ROC: {test_roc_auc:.4f}")

    # wandb.log({
    #     'test/acccuracy': test_accuracy,
    #     'test/loss': test_loss,
    #     'test/precision': test_precision,
    #     'test/recall': test_recall,
    #     'test/f1': test_f1,
    #     'test/roc_auc': test_roc_auc,
    # })

    end = time.time()
    Total_time=end-start
    print('Total training and inference time is: {0}'.format(Total_time))

    # run.finish()