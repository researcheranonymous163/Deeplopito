import torch.nn as nn
from torchvision import models
from transformers import BertModel
import torch
from collections import OrderedDict


resnet18 = models.resnet18(weights=None)
resnet18.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
resnet18.maxpool = nn.Identity()  # Remove maxpool
resnet18.fc = nn.Linear(resnet18.fc.in_features, 10)

resnet50 = models.resnet50(weights=None)
resnet50.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
resnet50.maxpool = nn.Identity()  # Remove maxpool
resnet50.fc = nn.Linear(resnet50.fc.in_features, 100)

# split cnn to 4 partition
cnn_partitions = [
    nn.Sequential(OrderedDict([
        ("part1_conv2", nn.Conv2d(1, 32, kernel_size=5)),
        ("part1_relu", nn.ReLU()),
        ("part1_maxPool2d", nn.MaxPool2d(kernel_size=2))
    ])),
    nn.Sequential(OrderedDict([
        ("part2_conv2", nn.Conv2d(32, 64, kernel_size=5)),
        ("part2_relu_part2", nn.ReLU()),
        ("part2_max_pool_part2", nn.MaxPool2d(kernel_size=2))
    ])),
    nn.Sequential(OrderedDict([
        ("part3_flatten", nn.Flatten()),
        ("part3_linear", nn.Linear(64 * 4 * 4, 512)),
        ("part3_relu", nn.ReLU())
    ])),
    nn.Sequential(OrderedDict([
        ("part4_linear", nn.Linear(512, 10))
    ]))
]
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



resnet_18_fully_layers_together = nn.Sequential(OrderedDict([
    ("part1_conv1", resnet18.conv1),
    ("part1_bn1", resnet18.bn1),
    ("part1_relu1", resnet18.relu),
    ("part1_pool", resnet18.maxpool),
    ("part2_layer", resnet18.layer1[0]),
    ("part3_layer", resnet18.layer1[0]),
    ("part4_layer", resnet18.layer2[0]),
    ("part5_layer", resnet18.layer2[1]),
    ("part6_layer", resnet18.layer3[0]),
    ("part7_layer", resnet18.layer3[1]),
    ("part8_layer", resnet18.layer4[0]),
    ("part9_layer", resnet18.layer4[1]),
    ("part10_avg_pool", resnet18.avgpool),
    ("part10_flatten", nn.Flatten()),
    ("part10_fc", resnet18.fc),
]))
resnet_18_all_layers = [
    nn.Sequential(OrderedDict([
        ("part1_conv1", resnet18.conv1),
        ("part1_bn1", resnet18.bn1),
        ("part1_relu1", resnet18.relu),
        ("part1_pool", resnet18.maxpool),
    ])),
    nn.Sequential(OrderedDict([
        ("part2_layer", resnet18.layer1[0]),
    ])),
    nn.Sequential(OrderedDict([
        ("part3_layer", resnet18.layer1[1]),
    ])),
    nn.Sequential(OrderedDict([
        ("part4_layer", resnet18.layer2[0]),
    ])),
    nn.Sequential(OrderedDict([
        ("part5_layer", resnet18.layer2[1]),
    ])),
    nn.Sequential(OrderedDict([
        ("part6_layer", resnet18.layer3[0]),
    ])),
    nn.Sequential(OrderedDict([
        ("part7_layer", resnet18.layer3[1]),
    ])),
    nn.Sequential(OrderedDict([
        ("part8_layer", resnet18.layer4[0]),
    ])),
    nn.Sequential(OrderedDict([
        ("part9_layer", resnet18.layer4[1]),
    ])),
    nn.Sequential(OrderedDict([
        ("part10_avg_pool", resnet18.avgpool),
        ("part10_flatten", nn.Flatten()),
        ("part10_fc", resnet18.fc),
    ]))
]



resnet_50_all_layers = [
    # Part 1: Initial stem (same as ResNet-18)
    nn.Sequential(OrderedDict([
        ("part1_conv1", resnet50.conv1),
        ("part1_bn1", resnet50.bn1),
        ("part1_relu", resnet50.relu),
    ])),

    # Layer1 (3 blocks)
    nn.Sequential(OrderedDict([("part2_layer1_0", resnet50.layer1[0])])),
    nn.Sequential(OrderedDict([("part3_layer1_1", resnet50.layer1[1])])),
    nn.Sequential(OrderedDict([("part4_layer1_2", resnet50.layer1[2])])),

    # Layer2 (4 blocks)
    nn.Sequential(OrderedDict([("part5_layer2_0", resnet50.layer2[0])])),
    nn.Sequential(OrderedDict([("part6_layer2_1", resnet50.layer2[1])])),
    nn.Sequential(OrderedDict([("part7_layer2_2", resnet50.layer2[2])])),
    nn.Sequential(OrderedDict([("part8_layer2_3", resnet50.layer2[3])])),

    # Layer3 (6 blocks)
    nn.Sequential(OrderedDict([("part9_layer3_0", resnet50.layer3[0])])),
    nn.Sequential(OrderedDict([("part10_layer3_1", resnet50.layer3[1])])),
    nn.Sequential(OrderedDict([("part11_layer3_2", resnet50.layer3[2])])),
    nn.Sequential(OrderedDict([("part12_layer3_3", resnet50.layer3[3])])),
    nn.Sequential(OrderedDict([("part13_layer3_4", resnet50.layer3[4])])),
    nn.Sequential(OrderedDict([("part14_layer3_5", resnet50.layer3[5])])),

    # Layer4 (3 blocks)
    nn.Sequential(OrderedDict([("part15_layer4_0", resnet50.layer4[0])])),
    nn.Sequential(OrderedDict([("part16_layer4_1", resnet50.layer4[1])])),
    nn.Sequential(OrderedDict([("part17_layer4_2", resnet50.layer4[2])])),

    # Part 18: Classifier
    nn.Sequential(OrderedDict([
        ("part18_avgpool", resnet50.avgpool),
        ("part18_flatten", nn.Flatten()),
        ("part18_fc", resnet50.fc)
    ]))
]
# Add this after resnet50_all_layers definition
resnet_50_fully_layers_together = nn.Sequential(
    nn.Sequential(OrderedDict([
        ("part1_conv1", resnet50.conv1),
        ("part1_bn1", resnet50.bn1),
        ("part1_relu", resnet50.relu)
    ])),
    nn.Sequential(OrderedDict([
        ("part2_layer1_0", resnet50.layer1[0]),
        ("part3_layer1_1", resnet50.layer1[1]),
        ("part4_layer1_2", resnet50.layer1[2])
    ])),
    nn.Sequential(OrderedDict([
        ("part5_layer2_0", resnet50.layer2[0]),
        ("part6_layer2_1", resnet50.layer2[1]),
        ("part7_layer2_2", resnet50.layer2[2]),
        ("part8_layer2_3", resnet50.layer2[3])
    ])),
    nn.Sequential(OrderedDict([
        ("part9_layer3_0", resnet50.layer3[0]),
        ("part10_layer3_1", resnet50.layer3[1]),
        ("part11_layer3_2", resnet50.layer3[2]),
        ("part12_layer3_3", resnet50.layer3[3]),
        ("part13_layer3_4", resnet50.layer3[4]),
        ("part14_layer3_5", resnet50.layer3[5])
    ])),
    nn.Sequential(OrderedDict([
        ("part15_layer4_0", resnet50.layer4[0]),
        ("part16_layer4_1", resnet50.layer4[1]),
        ("part17_layer4_2", resnet50.layer4[2])
    ])),
    nn.Sequential(OrderedDict([
        ("part18_avgpool", resnet50.avgpool),
        ("part18_flatten", nn.Flatten()),
        ("part18_fc", resnet50.fc)
    ])))

# ---------------- BERT ----------------
bert = BertModel.from_pretrained(
    "/root/.cache/huggingface/transformers/bert-base-uncased",
    local_files_only=True,
)

import os
NUM_LABELS = 4 if os.getenv("CURRENT_DATASET") == "AG_NEWS" else 2
classification_head = nn.Linear(bert.config.hidden_size, NUM_LABELS)

class BertEvalWrapper(nn.Module):
    def __init__(self, bert_model, classifier):
        super().__init__()
        self.bert = bert_model
        self.classifier = classifier

    def forward(self, input_ids, attention_mask=None):
        # Proper input handling with attention mask
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        # Use pooled output for classification
        pooled = outputs.pooler_output
        return self.classifier(pooled)

bert_fully_layers_together = BertEvalWrapper(bert, classification_head)

emb = bert.embeddings
enc_layers = list(bert.encoder.layer)
pooler = bert.pooler
pooler_classifier = nn.Sequential(
    pooler,
    classification_head
)

# 7. Partition list for distributed training
all_layers_bert = [emb] + enc_layers + [pooler_classifier]
