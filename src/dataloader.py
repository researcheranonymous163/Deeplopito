import logging
from random import Random

import torchvision
import torchvision.transforms as transforms
from dfl.data import stratified_split
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data.dataset import Subset
import torch
from torchvision.datasets import CIFAR10
from .constants import DATASET_COFAR10 , DATASET_MNIST , DATASET_IMDB , DATASET_AG_NEWS , DATASET_CIFAR100
import sys
import os
from config_folder import n_spit_dataset

dataset_name = os.environ.get("CURRENT_DATASET")
logging.critical(f"dataset name is :{dataset_name}")

class DataLoader:
    def __init__(self, batch_size=32
                 , shuffle=True, train_ratio=0.8,
                 index_client=1,d_name=dataset_name):
        """
        Initialize the DataLoader with specific configurations.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset_name = d_name
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.train_ratio = train_ratio
        self.vocab = None
        self.tokenizer = None

        if self.dataset_name == DATASET_MNIST:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))  # For single-channel images
            ])
        elif self.dataset_name == DATASET_COFAR10:
            self.train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2470, 0.2435, 0.2616])
            ])

            # Transformations for testing (less augmentation)
            self.test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2470, 0.2435, 0.2616])
                ])
        elif self.dataset_name == DATASET_CIFAR100:
            self.train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                     std=[0.2675, 0.2565, 0.2761])
            ])

            self.test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                     std=[0.2675, 0.2565, 0.2761])
            ])
        elif self.dataset_name in (DATASET_AG_NEWS,DATASET_IMDB):
            logging.critical("inside Ag news or imdb")
            from transformers import BertTokenizer
            self.tokenizer = BertTokenizer.from_pretrained("/root/.cache/huggingface/transformers/bert-base-uncased")
            self.collate_fn = self._bert_collate
        else:
            # fallback / or raise error
            raise ValueError(f"Unknown dataset name {self.dataset_name}")

        self.dataset_train, self.dataset_test = self.loadDataSet()
        self.train_split, self.test_split = self.stratified(index=index_client-1, n_splits=n_spit_dataset)
        logging.info(f"size of train dataset is {len(self.train_split)}")
        # Create train and test dataset

        if self.dataset_name==DATASET_MNIST:
            self.train_loader = TorchDataLoader(self.train_split, batch_size=self.batch_size, shuffle=self.shuffle)
            self.test_loader = TorchDataLoader(self.test_split, batch_size=self.batch_size, shuffle=False)
        elif self.dataset_name in (DATASET_COFAR10, DATASET_CIFAR100):
            logging.critical("loader cifar")
            self.train_loader = torch.utils.data.DataLoader(self.train_split, batch_size=batch_size, shuffle=True)
            self.test_loader = torch.utils.data.DataLoader(
                self.test_split, batch_size=batch_size, shuffle=False
            )
        else: # bert Model
            self.train_loader = TorchDataLoader(
                self.train_split ,
                batch_size=batch_size,
                shuffle=shuffle,
                collate_fn=self.collate_fn
            )
            self.test_loader = TorchDataLoader(
                self.test_split ,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=self.collate_fn
            )
        self.train_iter = iter(self.train_loader)
        self.test_iter = iter(self.test_loader)
        self.mode = "train"


    # use to sure every client have sample from all class
    def stratified(self, index: int, n_splits: int = 4):
        train_splits = stratified_split(
            list(range(len(self.dataset_train))),
            labels=self.dataset_train.targets,
            n_splits=n_splits,
            rnd=Random(1234)
        )

        test_splits = stratified_split(
            list(range(len(self.dataset_test))),
            labels=self.dataset_test.targets,
            n_splits=n_splits,
            rnd=Random(1234)
        )

        # Convert indices to Python integers
        train_indices = [int(i) for i in train_splits[index]]
        test_indices = [int(i) for i in test_splits[index]]

        return torch.utils.data.Subset(self.dataset_train, train_indices), torch.utils.data.Subset(
            self.dataset_test,
            test_indices)

    def set_mode(self, mode):
        """
        Set the mode to 'train' or 'test'.
        """
        if mode not in ["train", "test"]:
            raise ValueError("Mode must be either 'train' or 'test'")
        self.mode = mode
        if self.mode == "train":
            self.train_iter = iter(self.train_loader)
        else:
            self.test_iter = iter(self.test_loader)

    def loadDataSet(self):
        logging.critical("load dataset")

        if self.dataset_name == DATASET_MNIST:
            trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=self.transform)
            testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=self.transform)
        elif self.dataset_name==DATASET_COFAR10:
            trainset = torchvision.datasets.CIFAR10(
                root='./data', train=True,
                download=True, transform=self.train_transform
            )
            testset = torchvision.datasets.CIFAR10(
                root='./data', train=False,
                download=True, transform=self.test_transform
            )
        elif self.dataset_name == DATASET_CIFAR100:
            # Use the same transforms as CIFAR-10 or customize

            trainset = torchvision.datasets.CIFAR100(
                root='./data',
                train=True,
                download=True,  # Important: set to False since we have it
                transform=self.train_transform
            )
            testset = torchvision.datasets.CIFAR100(
                root='./data',
                train=False,
                download=True,  # Important: set to False
                transform=self.test_transform
            )
        elif self.dataset_name == DATASET_IMDB:
            from  datasets import load_dataset
            logging.critical("load data set using cache")
            dataset =load_dataset('imdb', cache_dir='/data/imdb')
            train_raw = dataset['train']
            test_raw = dataset['test']
            from .IMDBWrapper import IMDBDataset
            trainset = IMDBDataset(train_raw)
            testset=IMDBDataset(test_raw)

        elif self.dataset_name==DATASET_AG_NEWS:
            from datasets import load_dataset
            dataset = load_dataset('ag_news', cache_dir='/data/ag_news')
            train_raw, test_raw = dataset["train"], dataset["test"]

            class AGNewsDataset(torch.utils.data.Dataset):
                def __init__(self, split):
                    self.texts  = list(split["text"])
                    self.labels = [int(l) for l in split["label"]]
                    self.targets = self.labels

                def __len__(self): return len(self.labels)
                def __getitem__(self, idx):
                    return self.texts[idx], int(self.labels[idx])

            trainset, testset = AGNewsDataset(train_raw), AGNewsDataset(test_raw)
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")
        logging.critical("finish load dataset")
        return trainset, testset

    def getLoader(self):
        """
        Return the appropriate DataLoader based on the mode.
        """
        return self.train_loader if self.mode == "train" else self.test_loader

    def _bert_collate(self, batch):
        """Custom collate function for BERT tokenization"""
        texts, labels = zip(*batch)

        # Tokenize all texts in the batch
        encoding = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )

        # Return tuple format: (features, labels)
        return (encoding['input_ids'].to(self.device), encoding['attention_mask'].to(self.device)), torch.tensor(labels).to(self.device)
class DataManager:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.epoch = 0  # Track the current epoch
        self.current_iter = iter(self.data_loader.getLoader())  # Initialize iterator
        self.batch_count = 0
        self.epoch_done = False

    def next_batch(self):
        try:
            # Try to get the next batch
            batch = next(self.current_iter)
            self.batch_count += 1
        except StopIteration:
            # If StopIteration is raised, reset the iterator and increment epoch
            self.epoch += 1
            self.epoch_done = True
            self.batch_count = 1
            print(f"Epoch {self.epoch} completed.")
            self.current_iter = iter(self.data_loader.getLoader())
            batch = next(self.current_iter)  # Get the first batch of the new epoch

        return batch  # Return features and labels

    def reset_dataset(self):
        self.epoch = 0
        self.batch_count = 0
        self.current_iter = iter(self.data_loader.getLoader())
        self.epoch_done = False

# Usage example
if __name__ == "__main__":
    # Import and initialize the DataLoader for MNIST
    data_loader = DataLoader(dataset_name=DATASET_MNIST, batch_size=32, shuffle=True)

    # Set mode to 'train'
    data_loader.set_mode("train")

    # Initialize DataManager
    data_manager = DataManager(data_loader)

    # Set the maximum number of epochs
    max_epochs = 3

    # Loop over the dataset for the specified number of epochs
    while data_manager.epoch < max_epochs:
        features, labels = data_manager.next_batch()
        print(f"Features shape: {features.shape}, Labels shape: {labels.shape}")
        print(f"Current Epoch: {data_manager.epoch} current batch[{data_manager.batch_count}]")

    print("Finished processing dataset for 3 epochs.")
