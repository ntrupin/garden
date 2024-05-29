import argparse
import os
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

import mnist

class LeNet5(nn.Module):
    """
    CNN as defined in "Gradient-Based Learning Applied to Document Recognition"
    (LeCun et al, 1998). Takes 32x32 single-channel images as input and outputs
    10 probabilities.

    MNIST dataset provides 28x28 images. We add 2 pixels of padding to each side
    as LeNet expects 32x32.
    """

    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()

        self.extractor = nn.Sequential(
            # C1: 32x32x1 -> 28x28x6 convolution
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            # S2: 28x28x6 -> 14x14x6 subsampling
            nn.AvgPool2d(kernel_size=2, stride=2),

            # C3: 14x14x6 -> 16x16x10 convolution
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.ReLU(),
            # S4: 16x16x10 -> 5x5x16 subsampling
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            # C5: 400 -> 120 fully-connected
            nn.Linear(input_dims=16*5*5, output_dims=120),
            nn.ReLU(),

            # F6: 120 -> 84 fully-connected
            nn.Linear(input_dims=120, output_dims=84),
            nn.ReLU(),

            # Output: 84 -> 10 fully-connected
            nn.Linear(input_dims=84, output_dims=num_classes)
            #nn.LogSoftmax()
        )

    def __call__(self, x):
        # 784 -> 28x28x1
        x = x.reshape([-1, 28, 28, 1])
        # 28x28x1 -> 5x5x16
        x = self.extractor(x)
        # 5x5x16 -> 400
        x = x.reshape([-1, 5 * 5 * 16])
        # 400 -> 10
        return self.classifier(x)

PARAMS = {
    "batch_size": 256,
    "num_epochs": 10,
    "learning_rate": 0.01
}

def batch_iterate(batch_size, X, y):
    """
    group data (X) and labels (y) into batches of batch_size.
    randomly shuffle data between epochs.
    """
    perm = mx.array(np.random.permutation(y.size))
    for s in range(0, y.size, batch_size):
        ids = perm[s:s+batch_size]
        yield X[ids], y[ids]

def train(model):
    train_images, train_labels, test_images, test_labels = map(
        mx.array, getattr(mnist, "mnist")()
    )

    mx.eval(model.parameters())

    def loss_fn(model, X, y):
        return nn.losses.cross_entropy(model(X), y, reduction="mean")

    def eval_fn(X, y):
        return mx.mean(mx.argmax(model(X), axis=1) == y)

    optimizer = optim.SGD(learning_rate=PARAMS["learning_rate"], momentum=0.9)
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    for epoch in range(PARAMS["num_epochs"]):
        model.train()
        running_loss = 0.0

        tic = time.perf_counter()
        for X, y in batch_iterate(PARAMS["batch_size"], train_images, train_labels):
            loss, grads = loss_and_grad_fn(model, X, y)
            optimizer.update(model, grads)
            running_loss += loss.item()
        accuracy = eval_fn(test_images, test_labels)
        toc = time.perf_counter()
        print(f"Epoch {epoch + 1}, Loss {running_loss/PARAMS['batch_size']:.4f}, Accuracy {accuracy.item():.4f}, Time {toc - tic:.3f} (s)")
        running_loss = 0.0

    model.save_weights("lenet5.safetensors")

def main(args):
    model = LeNet5()
    if os.path.isfile(args.output):
        model.load_weights(args.output)
    else:
        train(model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train LeNet on MNIST with MLX.")
    parser.add_argument("-o", "--output", type=str, default="lenet5.safetensors",
                        help="output file name")
    parser.add_argument("--gpu", action="store_true", help="Use Metal backend.")
    args = parser.parse_args()

    if not args.gpu:
        mx.set_default_device(mx.cpu)

    main(args)
