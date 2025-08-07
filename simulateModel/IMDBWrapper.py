import torch
import numpy as np

class IMDBDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.texts = dataset['text']
        self.labels = dataset['label']
        # Convert labels to numpy array for stratified split compatibility
        self.targets = np.array(self.labels)  # Required for stratified split

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]