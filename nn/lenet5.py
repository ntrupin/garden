import argparse
from dataclasses import dataclass
import os
import time

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from einops.layers.torch import Rearrange, Reduce

import mnist

device = torch.device("mps")

class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.net = nn.Sequential(
            Rearrange("b (c h w) -> b c h w", c=1, h=28, w=28),
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),
            Rearrange("b c h w -> b (c h w)"),
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, num_classes)
        )

    def forward(self, x):
        return self.net(x)

@dataclass
class ModelArgs:
    batch_size: int = 256
    num_epochs: int = 10
    learning_rate: float = 0.01

def batch_iter(args: ModelArgs, X, y):
    perm = torch.randperm(y.size(dim=0)).to(device)
    for s in range(0, y.size(dim=0), args.batch_size):
        ids = perm[s:s+args.batch_size].to(torch.int32)
        yield X[ids], y[ids]

def train(model: LeNet5, args: ModelArgs, filename: str | None = None):
    train_images, train_labels, test_images, test_labels = map(
        torch.tensor, getattr(mnist, "mnist")()
    )
    train_images = train_images.to(device)
    train_labels = train_labels.long().to(device)
    test_images = test_images.to(device)
    test_labels = test_labels.long().to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)

    model.train()
    for epoch in range(args.num_epochs):

        running_loss = 0.0
        tic = time.perf_counter()
        for X, y in batch_iter(args, train_images, train_labels):
            loss = F.cross_entropy(model(X), y, reduction="mean")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        with torch.no_grad():
            accuracy = torch.mean(torch.eq(torch.argmax(model(test_images), dim=1), test_labels).float())
        toc = time.perf_counter()
        print(f"Epoch {epoch + 1}, Loss {running_loss/args.batch_size:.4f}, Accuracy {accuracy.item():.4f}, Time {toc - tic:.3f} (s)")
    model.eval()

    if filename is not None:
        torch.save(model.state_dict(), filename)

if __name__ == "__main__":
    model = LeNet5()
    model.to(device)
    if os.path.isfile("lenet5.pth"):
        model.load_state_dict(torch.load("lenet5.pth"))
    else:
        train(model, ModelArgs(), "lenet5.pth")
