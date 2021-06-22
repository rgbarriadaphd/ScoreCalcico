import math
import time
import logging
import os
import statistics
import torch
from PIL import Image, ImageOps
from torch.autograd.grad_mode import F
from torchvision import datasets, models, transforms
from data_selection import ScoreCalciumSelection
from hyperparams import *
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from utils import statistics



class ExtendedVGG(nn.Module):
    def __init__(self):
        super(ExtendedVGG, self).__init__()
        self.features = models.vgg16(pretrained=True).features
        self.avgpool = nn.AdaptiveAvgPool2d((self.features.in_features, self.features.in_features))
        self.classi
        self.cnn.fc = nn.Linear(
            self.cnn.fc.in_features, 20)

        self.fc1 = nn.Linear(20 + CLINICAL_DATA_DIM, 60)
        self.fc2 = nn.Linear(60, N_CLASSES)

    def forward(self, image, data):
        x1 = self.features(image)
        x2 = data

        x = torch.cat((x1, x2), dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = ExtendedVGG()

batch_size = 2
image = torch.randn(batch_size, 3, 299, 299)
data = torch.randn(batch_size, CLINICAL_DATA_DIM)

output = model(image, data)
