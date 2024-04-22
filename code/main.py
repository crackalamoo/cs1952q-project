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

EPOCHS = int(args.epochs) if args.epochs is not None else 20
device = args.device or "cpu"
USE_CURRICULUM = True
LOSS_THRESHOLD = 0.8
REINTRODUCE_LEARNED = 0.1
STORED_LOSSES = 3
LR = 2e-3

def load_curriculum(train_loader, sample_losses, batch_size=None):
    if batch_size is None: batch_size = train_loader.batch_size

    dataset = train_loader.dataset
    keep_idx = set()
    for idx in sample_losses:
        if np.mean(sample_losses[idx]) > LOSS_THRESHOLD or np.random.random() < REINTRODUCE_LEARNED:
            keep_idx.add(idx)
    subset = Subset(dataset, list(keep_idx))
    proportion = len(keep_idx)/(len(train_loader)*batch_size)
    print("Proportion kept:", proportion)

    training = DataLoader(subset, batch_size=batch_size, shuffle=True)
    return training, proportion

def train(model, train_loader, val_loader=None, epochs=EPOCHS, use_curriculum=USE_CURRICULUM, lr=LR):
    losses = []
    accuracies = []
    val_losses = []
    val_accuracies = []
    training = train_loader
    sample_losses = defaultdict(lambda: 900*np.ones(STORED_LOSSES))
    times = []
    proportions = []
    start_time = time.time()
    for epoch in range(epochs):
        model.to(device)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        accuracy = []
        epoch_losses = []
        if use_curriculum and epoch >= STORED_LOSSES:
            training, proportion = load_curriculum(train_loader, sample_losses)
            proportions.append(proportion)
        else:
            proportions.append(1)
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
        return {
            'loss': losses, 'acc': accuracies, 'val_loss': val_losses, 'val_acc': val_accuracies,
            'times': times, 'prop': proportions
        }
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
    train_res = train(model, train_loader, val_loader=test_loader, epochs=EPOCHS)
    train_loss, train_acc, val_loss, val_acc, train_times, proportions = (train_res['loss'],
        train_res['acc'], train_res['val_loss'], train_res['val_acc'],
        train_res['times'], train_res['prop'])

    test_loss, test_acc = test(model, test_loader)

    print(f"Final Test Accuracy: {test_acc}")
    print(f"Total time: {train_times[-1]} s")

    plt.figure()
    if USE_CURRICULUM:
        plt.title('Dynamic curriculum learning avoids overfitting on CIFAR-10')
    else:
        plt.title('Model performance without dynamic curriculum')
    plt.plot(train_times, train_loss, label='training loss')
    plt.plot(train_times, val_loss, label='validation loss')
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Cross-entropy loss')
    if USE_CURRICULUM:
        plt.savefig('../results/cur_train_val.png')
    else:
        plt.savefig('../results/reg_train_val.png')

    plt.figure()
    plt.plot(train_times, train_acc, label='train accuracy')
    plt.plot(train_times, val_acc, label='validation accuracy')
    plt.plot(train_times[STORED_LOSSES:], proportions[STORED_LOSSES:], label='proportion')
    plt.xlabel("Time (s)")
    plt.ylabel("Accuracy/proportion")
    plt.title("Accuracy and training proportion over time")
    if USE_CURRICULUM:
        plt.savefig('../results/cur_acc.png')
    else:
        plt.savefig('../results/reg_acc.png')

    fname = 'cur' if USE_CURRICULUM else 'reg'
    with open(f'../results/{fname}.npy', 'wb') as f:
        np.save(f, train_times)
        np.save(f, val_loss)
        np.save(f, val_acc)
    
    if USE_CURRICULUM:
        try:
            with open(f'../results/reg.npy', 'rb') as f:
                reg_times = np.load(f)
                reg_loss = np.load(f)
                reg_acc = np.load(f)
            plt.figure()
            plt.plot(train_times, val_acc, label='curriculum')
            plt.plot(reg_times, reg_acc, label='no curriculum')
            plt.plot(train_times[STORED_LOSSES:], proportions[STORED_LOSSES:], label='proportion')
            plt.xlabel('Time (s)')
            plt.ylabel('Accuracy/proportion')
            plt.title('Faster convergence and comparable performance on CIFAR-10')
            plt.legend()
            plt.savefig('../results/cur_vs_reg.png')

            plt.figure()
            plt.plot(train_times, val_loss, label='curriculum')
            plt.plot(reg_times, reg_loss, label='no curriculum')
            plt.xlabel('Time (s)')
            plt.ylabel('Cross-entropy loss')
            plt.title('Faster convergence and comparable performance on CIFAR-10')
            plt.legend()
            plt.savefig('../results/cur_vs_reg_loss.png')
        except FileNotFoundError: pass

    plt.show()


if __name__ == '__main__':
    main()