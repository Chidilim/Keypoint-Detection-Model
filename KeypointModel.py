import os
import fnmatch
import torch
from torch.utils.data import Dataset
from torchvision import transforms, models
import torchvision
import cv2
import numpy as np
import torch.nn as nn



class KeypointModel(nn.Module):
    def __init__(self):
        super(KeypointModel, self).__init__()
        self.resnet = torchvision.models.resnet18(pretrained=True)
        self.global_avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.fc_new = nn.Linear(in_features=1000, out_features=2)  
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.resnet(x)
        x = x.view(x.size(0), -1)
        x = self.fc_new(x)
        return x
