import sys
import datetime
import torch.nn as nn
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision.models as models
import argparse
from OxfordDataset2 import OxfordDataset2
import time
import random
from KeypointModel import KeypointModel

# Maahum Khan, Dilly Ejeh — ELEC475 Lab 5


import os
import cv2

def readLineAt(file_path, index):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        if 0 <= index < len(lines):
            return lines[index].strip()
        else:
            return None

def visualize(images_folder, labels_file, pred_file, random_index):
    image_files = [f for f in os.listdir(images_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

    firstLabels = readLineAt(labels_file, random_index)
    if firstLabels is not None:
        image_name, label_str = firstLabels.split(',', 1)
        label = tuple(map(int, label_str.strip('()"').split(',')))

    firstPred = readLineAt(pred_file, random_index)
    if firstPred is not None:

        coordinates = tuple(map(int, firstPred.strip('()').split(',')))

    image_data = []
    for image_file in image_files:

        if image_name in image_file:
                image_path = os.path.join(images_folder, image_file)
                image = cv2.imread(image_path)

                cv2.circle(image, label, 10, (0, 255, 0), -1)
                cv2.circle(image, coordinates, 10, (255, 0, 0), -1)

                image_data.append((f"Image {len(image_data) + 1}", image, label))

    cv2.imshow(image_name, image)


    cv2.waitKey(0)
    cv2.destroyAllWindows()



def test(model, dataloader, device, output_file = "pred_output.txt"):
    starttime = time.time()
    pred_values = []
    label_values = []
    numImages = 0
    model.eval()
    with torch.no_grad(), open(output_file, 'w') as pred_file:
        for images, labels in dataloader:
            images = images.to(device=device)
            labels = labels.to(device=device)
            numImages += labels.size(0)
            output = model(images)


            output_numpy = output.cpu().numpy()
            labels_numpy = labels.cpu().numpy()

            for i in range(len(images)):

                pred_coords = f'({int(output_numpy[i][0])}, {int(output_numpy[i][1])})'

                pred_file.write(f"{pred_coords}\n")



            pred_values.extend(output_numpy)
            label_values.extend(labels_numpy)


    endtime = time.time()

    elapsedtime = ((endtime - starttime) * 1000) / numImages

    print(f"Inference took : {elapsedtime} milliseconds per image to complete ")

    pred_values = np.array(pred_values)
    label_values = np.array(label_values)


    mae_value = np.mean(np.abs(pred_values - label_values))

    print("The mean absolute error for this model is ", mae_value)





if __name__ == "__main__":
    # Argument passing
    numImages = 0
    parser = argparse.ArgumentParser()

    parser.add_argument("-s", type=str, default="model.pth", help="Decoder Path")
    parser.add_argument("-l", type=str, default="./", help="Labels file path")
    parser.add_argument("-i", type=str, default="./", help="Images file path")

    args = parser.parse_args()

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomRotation(degrees=(-30, 30)),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    label_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    test_data = OxfordDataset2(args.l, args.i, training=False, transform=transform, label_transform=label_transform)

    keypointmodel = KeypointModel()
    device = torch.device("cpu")
    keypointmodel.to(device)
    state_dict = torch.load(args.s, map_location=torch.device('cpu'))
    keypointmodel.load_state_dict(state_dict)

    TestLoader = DataLoader(test_data, batch_size=100, shuffle=False)

    test(keypointmodel, TestLoader, device)


    visualize(args.i, "test_noses.txt", "pred_output.txt",10)
    visualize(args.i, "test_noses.txt", "pred_output.txt", 200)
    visualize(args.i, "test_noses.txt", "pred_output.txt", 100)




    #python3 test.py -s model.pth -l ./ -i  ./data/images/
