from __future__ import absolute_import

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from cifar10 import Cifar10Model, get_cifar10_data
import matplotlib.pyplot as plt
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("--force-train", action=argparse.BooleanOptionalAction)
parser.add_argument("--device")
parser.add_argument("--epochs")
args = parser.parse_args()

FORCE_TRAIN = args.force_train or False
EPOCHS = int(args.epochs) if args.epochs is not None else 10
device = args.device or "cpu"
LOSS_THRESHOLD = 0.1

def load_curriculum(train_loader, sample_losses, batch_size=None):
    if batch_size is None: batch_size = train_loader.batch_size

    dataset = train_loader.dataset
    keep_idx = set()
    learned_idx = set()
    for idx in sample_losses:
        if np.mean(sample_losses[idx]) > 1e-1:
            keep_idx.add(idx.item())
        else:
            learned_idx.add(idx.item())
    subset = Subset(dataset, list(keep_idx))

    training = DataLoader(subset, batch_size=batch_size, shuffle=True)
    return training, learned_idx

def train(model, train_loader, val_loader=None, epochs=EPOCHS, use_curriculum=True, lr=1e-3):
    '''
    Trains the model on all of the inputs and labels.

    :param model: the initialized model to use for the forward pass and backward pass
    :param train_inputs: train inputs (all inputs to use for training), 
    shape (num_inputs, width, height, num_channels)
    :param train_labels: train labels (all labels to use for training), 
    shape (num_labels, num_classes)
    :return: Return the average accuracy across batches of the train inputs/labels
    '''
    losses = []
    accuracies = []
    val_losses = []
    val_accuracies = []
    training = train_loader
    learned = set()
    sample_losses = defaultdict(lambda: 900*np.ones(5))
    for epoch in range(epochs):
        model.to(device)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        accuracy = []
        epoch_losses = []
        if use_curriculum and epoch > 0:
            training, learned = load_curriculum(train_loader, sample_losses)
        for inputs, labels, idxs in training:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            batch_losses = torch.nn.functional.cross_entropy(outputs, labels.float(), reduction='none')
            for idx, loss_i in zip(idxs, batch_losses):
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
    if val_loader is not None:
        return losses, accuracies, val_losses, val_accuracies
    return losses, accuracies

def test(model, test_loader):
    """
    Tests the model on the test inputs and labels.

    :param model: hi
    :param test_inputs: test data (all images to be tested), 
    shape (num_inputs, width, height, num_channels)
    :param test_labels: test labels (all corresponding labels),
    shape (num_labels, num_classes)
    :return: test accuracy - this should be the average accuracy across
    all batches
    """
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

def load_or_train_model(ModelClass, train_loader, save_path, epochs=10, force_train=False):
    model = ModelClass()
    def do_train():
        train(model, train_loader, epochs=epochs)
        torch.save(model.state_dict(), save_path)
    if force_train: do_train()
    else:
        try:
            model.load_state_dict(torch.load(save_path))
            print("LOADED MODEL")
        except FileNotFoundError:
            do_train()
    return model


def main():
    '''
    Read in CIFAR10 data (limited to 2 classes), initialize your model, and train and 
    test your model for a number of epochs. We recommend that you train for
    10 epochs and at most 25 epochs.

    Consider printing the loss, training accuracy, and testing accuracy after each epoch
    to ensure the model is training correctly.

    CS1470 students should receive a final accuracy 
    on the testing examples for cat and dog of >=70%.

    CS2470 students should receive a final accuracy 
    on the testing examples for cat and dog of >=75%.

    :return: None
    '''
    train_loader, test_loader = get_cifar10_data()

    model = Cifar10Model()
    train_loss, train_acc, val_loss, val_acc = train(model, train_loader, val_loader=test_loader, epochs=EPOCHS)

    test_loss, test_acc = test(model, test_loader)

    print(f"Final Test Accuracy: {test_acc}")

    plt.figure()
    plt.plot(np.arange(len(train_loss)), train_loss)
    plt.plot(np.arange(len(val_loss)), val_loss)

    plt.figure()
    plt.plot(np.arange(len(train_acc)), train_acc)
    plt.plot(np.arange(len(val_acc)), val_acc)

    plt.show()


if __name__ == '__main__':
    main()