import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
import os
import zipfile
from sklearn.model_selection import train_test_split
from PIL import Image
import pandas as pd
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler

# Replace 'YOUR_ENV_VARIABLE' with the name of your environment variable
env_variable_name = 'PYTHONPATH'

# Get the value of the environment variable
env_variable_value = os.environ.get(env_variable_name)

# Check if the environment variable exists and print its value
if env_variable_value is not None:
    print(f'The value of {env_variable_name} is: {env_variable_value}')
else:
    print(f'The environment variable {env_variable_name} is not set.')


class TurtleDataset(Dataset):

    def __init__(self, csv_file='../data_imbalance/turtle_image_metadata_clean_s.csv', transform=None, sampling_strategy=None, method=None, test_size: float = 0.2, train=True, random_state=42):
    # def __init__(self, csv_file='../data_imbalance/turtle_image_metadata_r1_partition.csv', transform=None, sampling_strategy=None, method=None, test_size: float = 0.2, train=True, random_state=42):

        self.df = pd.read_csv(f"./data_imbalance/{csv_file}")
        # self.df = pd.read_csv(f"./{csv_file}")

        self.is_train = train

        # Define data transformations
        self.transform = transform

        # Split the dataset into train and test
        train_indices, test_indices = train_test_split(
            range(len(self.df)),
            test_size=test_size,
            random_state=random_state
        )

        self.indices = train_indices if train else test_indices

    def get_labels(self):
        # Return the train data
        return [(label if label == '0' else '1') for label in self.df.iloc[self.indices]['label'].to_list()]
    
    def get_labels_from_indices(self, indices):

        return [(label if label == '0' else '1') for label in self.df.iloc[indices]['label'].to_list()]

    def get_test_data(self):
        # Return the test data
        return self.test_data    

    def __len__(self):
        return len(self.df.iloc[self.indices])

    def __getitem__(self, idx):

        idx = self.indices[idx]

        img_folder = self.df.iloc[idx, 0]
        img_name = self.df.iloc[idx, 1]
        # img_path = f"../{img_folder}/{img_name}"
        img_path = f"{img_folder}/{img_name}"

        image = Image.open(img_path).convert("RGB")
 
        label = int(self.df.iloc[idx, 6] if self.df.iloc[idx, 6] != 'Certain Turtle' else '1')
        top = int(self.df.iloc[idx, 4])
        left = int(self.df.iloc[idx, 5])

        if self.transform:
            image = self.transform[int(label)](image) if self.is_train else self.transform(image)   

        return image, label, top, left
