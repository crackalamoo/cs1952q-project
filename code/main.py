from __future__ import absolute_import

import time
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from cifar10 import Cifar10Model, get_cifar10_data
import matplotlib.pyplot as plt
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("--device")
parser.add_argument("--epochs")
args = parser.parse_args()

EPOCHS = int(args.epochs) if args.epochs is not None else 10
device = args.device or "cpu"
LOSS_THRESHOLD = 1.0
REINTRODUCE_LEARNED = 0.1
STORED_LOSSES = 3
LR = 1e-3

def load_curriculum(train_loader, sample_losses, batch_size=None):
    if batch_size is None: batch_size = train_loader.batch_size

    dataset = train_loader.dataset
    keep_idx = set()
    for idx in sample_losses:
        if np.mean(sample_losses[idx]) > LOSS_THRESHOLD or np.random.random() < REINTRODUCE_LEARNED:
            keep_idx.add(idx)
    subset = Subset(dataset, list(keep_idx))
    print("Proportion kept:", len(keep_idx)/len(train_loader))

    training = DataLoader(subset, batch_size=batch_size, shuffle=True)
    return training

def train(model, train_loader, val_loader=None, epochs=EPOCHS, use_curriculum=True, lr=LR):
    losses = []
    accuracies = []
    val_losses = []
    val_accuracies = []
    training = train_loader
    sample_losses = defaultdict(lambda: 900*np.ones(STORED_LOSSES))
    start_time = time.time()
    times = []
    for epoch in range(epochs):
        model.to(device)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        accuracy = []
        epoch_losses = []
        if use_curriculum and epoch >= STORED_LOSSES:
            training = load_curriculum(train_loader, sample_losses)
        for inputs, labels, idxs in training:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            batch_losses = torch.nn.functional.cross_entropy(outputs, labels.float(), reduction='none')
            for idx, loss_i in zip(idxs, batch_losses):
                idx = idx.item()
                sample_losses[idx] = np.roll(sample_losses[idx], -1)
                sample_losses[idx][-1] = loss_i.item()
            loss = torch.mean(batch_losses)
            accuracy.append(model.accuracy(outputs, labels))
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        train_acc = np.mean(accuracy)
        accuracies.append(train_acc)
        losses.append(np.mean(epoch_losses))
        if val_loader is not None:
            val_loss, val_acc = test(model, val_loader)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)
            print(f"Epoch: {epoch}, Training Accuracy: {train_acc}, Validation Accuracy: {val_acc}")
        else:
            print(f"Epoch: {epoch}, Training Accuracy: {train_acc}")
        times.append(time.time() - start_time)
    if val_loader is not None:
        return losses, accuracies, val_losses, val_accuracies, times
    return losses, accuracies, times

def test(model, test_loader):
    model.to("cpu")
    model.eval()
    losses = []
    accuracy = []
    with torch.no_grad():
        for inputs, labels, idxs in test_loader:
            inputs, labels = inputs, labels
            outputs = model(inputs)
            losses.append(model.loss(outputs, labels))
            accuracy.append(model.accuracy(outputs, labels))
    test_acc = np.mean(accuracy)
    test_loss = np.mean(losses)
    return test_loss, test_acc

def main():
    train_loader, test_loader = get_cifar10_data()

    model = Cifar10Model()
    train_loss, train_acc, val_loss, val_acc, train_times = train(model, train_loader, val_loader=test_loader, epochs=EPOCHS)

    test_loss, test_acc = test(model, test_loader)

    print(f"Final Test Accuracy: {test_acc}")

    plt.figure()
    plt.plot(train_times, train_loss)
    plt.plot(train_times, val_loss)

    plt.figure()
    plt.plot(train_times, train_acc)
    plt.plot(train_times, val_acc)

    plt.show()


if __name__ == '__main__':
    main()