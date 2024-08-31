
# Maahum Khan, Dilly Ejeh — ELEC475 Lab 5
# Dataset — Adapted from YODADataset.py provided by Dr. Greenspan from Lab 4

import os
import fnmatch
import torch
from torch.utils.data import Dataset
from torchvision import transforms, models
import cv2
import numpy as np



class OxfordDataset2(Dataset):
    def __init__(self, label_dir, img_dir, training, transform, label_transform):
        self.dir = dir
        self.training = training
        self.mode = 'train'
        self.labels_file = 'train_noses.3.txt'
        if self.training == False:
            self.mode = 'test'
            self.labels_file = 'test_noses.txt'

        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.label_transform = label_transform
        self.num = 0
        label_path = os.path.join(self.label_dir, self.labels_file)
        with open(label_path) as file:
            lines = file.readlines()

        self.labels = []
        self.img_for_label = []
        for line in lines:
            self.labels.append(line)
            # Put name of corresponding img file in array
            parts = line.strip().split(",")
            self.img_for_label.append(parts[0])

        # Create array of images
        self.img_files = []
        for file in self.img_for_label:
            if fnmatch.fnmatch(file, '*.jpg'):
                self.img_files += [file]

        self.max = len(self)


    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        filename = os.path.splitext(self.img_files[idx])[0]
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
          image = cv2.resize(image, (224, 224))



        image = self.transform(image)
        # Get key point coordinates
        coords = self.labels[idx].split('"(')[1].strip()
        coords = coords.rstrip(')"')

        coords_string = coords.split(',')
        x = coords_string[0].strip()
        y = coords_string[1].strip()





        label = [float(x), float(y)]
        label_tensor = torch.tensor(label)

        return image, label_tensor

    def __iter__(self):
        self.num = 0
        return self


    def __next__(self):
        if (self.num >= self.max):
            raise StopIteration
        else:
            self.num += 1
            return self.__getitem__(self.num - 1)

