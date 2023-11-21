import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import os
import zipfile
from sklearn.model_selection import train_test_split

# class TurtleDataset(Dataset):
#     def __init__(self, zip_path, transform=None, train=True, test_size=0.2):
#         self.zip_path = zip_path
#         self.transform = transform
#         self.extracted_folder = self.extract_zip()

#         # Assuming the zip file structure is such that each class has a separate folder
#         all_data = datasets.ImageFolder(
#             root=self.extracted_folder,
#             transform=self.transform
#         )

#         # Split the data into train and test sets
#         train_indices, test_indices = train_test_split(
#             list(range(len(all_data))),
#             test_size=test_size,
#             random_state=42
#         )

#         self.dataset = all_data if train else torch.utils.data.Subset(all_data, test_indices)

#     def extract_zip(self):
#         extract_to = 'extracted_dataset'
#         if not os.path.exists(extract_to):
#             with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
#                 zip_ref.extractall(extract_to)
#         return extract_to

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, idx):
#         return self.dataset[idx]

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd

class TurtleDataset(Dataset):

    def __init__(self, csv_file='turtle_image_metadata.csv', transform=None, test_size: float = 0.2, train=True):

        self.df = pd.read_csv(f"./{csv_file}")
        self.do_transform = transform

        # Define data transformations
        self.transform = transforms.Compose([
            # transforms.Resize((120, 120)),  # Adjust the size as needed
            transforms.ToTensor(),
        ])

        # Split the dataset into train and test
        train_indices, test_indices = train_test_split(
            range(len(self.df)),
            test_size=test_size,
            random_state=42
        )

        self.indices = train_indices if train else test_indices

    def get_train_data(self):
        # Return the train data
        return self.train_data

    def get_test_data(self):
        # Return the test data
        return self.test_data    

    def __len__(self):
        return len(self.df.iloc[self.indices])

    def __getitem__(self, idx):

        idx = self.indices[idx]

        img_folder = self.df.iloc[idx, 0]
        img_name = self.df.iloc[idx, 1]
        img_path = f"../{img_folder}/{img_name}"

        image = Image.open(img_path).convert("RGB")

        if self.do_transform:
            image = self.transform(image)

        label = int(self.df.iloc[idx, 6] if self.df.iloc[idx, 6] != 'Certain Turtle' else '2')

        return image, label
