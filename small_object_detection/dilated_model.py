import torch.nn as nn
import torch.nn.functional as F
import torchvision

class DilatedCNNObjectDetection(nn.Module):

    def __init__(self, num_classes=2, num_boxes=1):
    
        super(DilatedCNNObjectDetection, self).__init__()

        # Convolutional Layers with Dilation
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, dilation=2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=2, dilation=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=2, dilation=2)

        
        # Fully Connected Layers for Classification
        self.fc_cls1 = nn.Linear(64 * 64 * 15 * 15, 1024)
        self.fc_cls2 = nn.Linear(1024, 1024)
        self.fc_cls3 = nn.Linear(1024, num_classes)

        # Fully Connected Layers for Bounding Box Regression
        self.fc_bbox1 = nn.Linear(64 * 64 * 15 * 15, 128)
        self.fc_bbox2 = nn.Linear(128, 64)
        self.fc_bbox3 = nn.Linear(64, 32)
        self.fc_bbox4 = nn.Linear(32, 2 * num_boxes)  # Assuming top-left corner of the box

        # Activation and Dropout Layers
        self.activation = nn.ReLU()
        self.activation_regression = nn.Sigmoid()
        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.5)

    def forward(self, x):
        # Convolutional Layer Block 1
        x1 = self.activation(self.conv1(x))
        # Convolutional Layer Block 2
        x2 = self.activation(self.conv2(x1))
        # Convolutional Layer Block 3
        x3 = self.activation(self.conv3(x2))

        # Flatten4
        x_cls = x3.view(-1, 64 * 64 * 15 * 15)
        x_bbox = x3.view(-1, 64 * 64 * 15 * 15)

        # Fully Connected Layers for Classification
        x_cls = self.dropout1(self.activation(self.fc_cls1(x_cls)))
        x_cls = self.dropout2(self.activation(self.fc_cls2(x_cls)))
        x_cls = self.fc_cls3(x_cls)

        # Fully Connected Layers for Bounding Box Regression
        x_bbox = self.fc_bbox3(self.activation(self.fc_bbox2(self.activation(self.fc_bbox1(x_bbox)))))
        x_bbox = self.activation_regression(self.fc_bbox4(self.activation(x_bbox)))

        # Apply softmax for classification
        probs = F.softmax(x_cls, dim=1)

        return probs, x_bbox.view(-1, 1, 2)  # Reshape to (batch_size, num_boxes, 2)

    def count_parameters(self):
        total_params = 0
        for name, param in self.named_parameters():
            num_params = param.numel()
            total_params += num_params
            print(f"{name}: {num_params} parameters")
        print(f"Total number of parameters in the model: {total_params}")
        return total_params
    
    def calculate_loss(self, preds_cls, targets_cls, preds_bbox, targets_bbox, lambda_coord=5.0):
        # Classification loss
        cls_loss = F.cross_entropy(preds_cls, targets_cls)

        # Bounding box regression loss (Smooth L1 loss)
        bbox_loss = F.smooth_l1_loss(preds_bbox.reshape(targets_bbox.shape), targets_bbox, reduction='sum')

        # Total loss
        total_loss = cls_loss + lambda_coord * bbox_loss

        return total_loss

    def non_max_suppression(self, boxes, scores, iou_threshold=0.5):
        # Apply non-maximum suppression to filter out overlapping bounding boxes
        keep = torchvision.ops.nms(boxes, scores, iou_threshold)
        return keep