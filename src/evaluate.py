import logging
import torch
import torch.nn as nn
import torch.utils.data
from torchvision import datasets, transforms , models
from collections import OrderedDict
from .logger import write_accuracy
from .train import DataLoader , DataManager
import os
from .constants import MODEL_CNN , MODEL_RESNET18 , MODEL_BERT , MODEL_RESNET50

model_name = os.environ.get("CURRENT_MODEL")

resnet = models.resnet18(weights=None)  # not pretrained
in_features = resnet.fc.in_features
resnet.fc = nn.Linear(in_features, 10)
    

# from baseModel import resnet_layers , cnn_layers
model_name = os.environ.get("CURRENT_MODEL")

resnet = models.resnet18(weights=None)  # not pretrained
in_features = resnet.fc.in_features
resnet.fc = nn.Linear(in_features, 10)


cnn_layers = nn.Sequential(OrderedDict([
    ("part1_conv2", nn.Conv2d(1, 32, kernel_size=5)),
    ("part1_relu", nn.ReLU()),
    ("part1_maxPool2d", nn.MaxPool2d(kernel_size=2)),
    ("part2_conv2", nn.Conv2d(32, 64, kernel_size=5)),
    ("part2_relu_part2", nn.ReLU()),  # Changed from MaxPool2d to ReLU
    ("part2_max_pool_part2", nn.MaxPool2d(kernel_size=2)),
    ("part3_flatten", nn.Flatten()),
    ("part3_linear", nn.Linear(64 * 4 * 4, 512)),
    ("part4_linear", nn.Linear(512, 10))
]))

from baseModel import  resnet_18_fully_layers_together, cnn_layers,bert_fully_layers_together ,resnet_50_fully_layers_together

class FullModel(nn.Module):
    def __init__(self, state_dict):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_name = os.environ.get("CURRENT_MODEL")

        if model_name == MODEL_RESNET18:
            logging.critical("ResNet18 selected")
            self.layers = resnet_18_fully_layers_together
        elif model_name == MODEL_RESNET50:
            self.layers = resnet_50_fully_layers_together
        elif model_name == MODEL_CNN:
            logging.critical("CNN selected")
            self.layers = cnn_layers
        elif model_name == MODEL_BERT:
            logging.critical("BERT selected")
            self.layers = bert_fully_layers_together
        else:
            raise ValueError(f"Unknown model name {model_name}")
        self.layers = self.layers.to(self.device)


        logging.critical(f"state dict in evaluate function is {state_dict.keys()}")
        if model_name != MODEL_BERT:
            flat_state_dict = {}
            for partition_name , partition_state in state_dict.items():
                for key , param in partition_state.items():
                    flat_state_dict[key] = param
                    self.layers.load_state_dict(flat_state_dict,strict=False)
        else:
            self.layers.load_state_dict(state_dict, strict=False)
        logging.critical(f"Loaded params: {len(self.layers.state_dict())}")

    def forward(self, *args, **kwargs):
        return self.layers(*args, **kwargs)

    def evaluate(self, test_loader):
        self.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in test_loader:
                # Handle BERT's special format
                if model_name == MODEL_BERT:
                    # Unpack the tuple structure
                    (input_ids, attention_mask), labels = batch

                    input_ids = input_ids.to(self.device)
                    attention_mask = attention_mask.to(self.device)
                    labels = labels.to(self.device)

                    outputs = self.forward(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )

                # Handle CNN/ResNet format
                elif isinstance(batch, (list, tuple)) and len(batch) == 2:
                    inputs, labels = batch
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.forward(inputs)

                else:
                    raise ValueError("Unrecognized batch format")

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100.0 * correct / total
        logging.info(f"Test Accuracy: {accuracy:.2f}%")
        write_accuracy(accuracy)
        return accuracy