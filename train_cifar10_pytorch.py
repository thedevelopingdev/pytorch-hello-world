#!/usr/bin/env python

"""
Trains a ResNet50 model on the CIFAR10 data set.

source(s): 
1. https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
"""

import argparse

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

from tqdm import tqdm
from resnet import ResNet50

def main(device):
    # load data
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 128

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # initialize a neural network
    net = ResNet50()
    net = net.to(device)

    # define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    INSTANCES_PER_PRINT_STATS = 8000
    print_interval = int(INSTANCES_PER_PRINT_STATS / batch_size)

    # train
    for epoch in range(10):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in tqdm(enumerate(trainloader, 0)):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % print_interval == print_interval - 1:    # print every 8000 instances
                print(f'[{epoch + 1}, minibatch {i + 1:5d}] loss: {running_loss / print_interval:.3f}')
                running_loss = 0.0

    print('Finished Training')


def get_default_device():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",
                        default=get_default_device())
    args = parser.parse_args()

    print(f"Using device: {args.device}")

    main(device=args.device)
