import sys
import datetime
import os
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
from KeypointModel import KeypointModel


def train(model, optimizer, train_loader, device, numEpochs, loss_fn, weightpth, loss_plot):
    losses_train = []
    model.train()
    losses_val = []

    print("Entering epoch")
    for epoch in range(1, numEpochs + 1):
        loss_train = 0.0
        count = 0
        for images, labels in train_loader:

            count += 1
            images = images.to(device=device)
            labels = labels.to(device=device)
            optimizer.zero_grad()
            output = model(images)

            loss = loss_fn(output, labels)

            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
            loss_train += loss.item()

            # (For our personal limited GPU memory)
            if device == "cuda":
                torch.cuda.empty_cache()


        losses_train += [loss_train / len(train_loader)]

        print(f'{datetime.datetime.now()} Epoch {epoch}, TRAINING loss {(loss_train / len(train_loader))}')

    plt.figure(figsize=(12, 7))
    plt.plot(losses_train, label='Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc=1)
    plt.title('Training Loss Over Time')
    plt.savefig(loss_plot)
    plt.show()
    torch.save(model.state_dict(), weightpth)
    return losses_train


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", type=int, default=50, help="Epochs")
    parser.add_argument("-b", type=int, default=4, help="Batch size")
    parser.add_argument("-s", type=str, default="model.pth", help="Decoder Path")
    parser.add_argument("-plot", type=str, default="loss.png", help="Loss png")
    parser.add_argument("-l", type=str, default="./", help="Labels file path")
    parser.add_argument("-i", type=str, default="./data/images/", help="Images file path")

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


    training_data = OxfordDataset2(args.l, args.i, training=True, transform=transform, label_transform=label_transform)


    keypointmodel = KeypointModel()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    keypointmodel.to(device)

    learning_rate = 1e-3
    loss_fn = nn.SmoothL1Loss()

    optimizer = optim.Adam(keypointmodel.parameters(), lr=learning_rate)

    train_loader = DataLoader(training_data, batch_size=args.b, shuffle=True)

    train(keypointmodel, optimizer, train_loader, device, args.e, loss_fn, args.s, args.plot)





