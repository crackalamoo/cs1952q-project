from __future__ import absolute_import

import argparse
import numpy as np
import torch
from preprocess import get_data
from cifar10 import Cifar10Model
from visuals import save_tensor_gifs

parser = argparse.ArgumentParser()
parser.add_argument("--force-train", action=argparse.BooleanOptionalAction)
parser.add_argument("--device")
args = parser.parse_args()

FORCE_TRAIN = args.force_train or False
device = args.device or "cpu"


def train(model, train_loader, lr=1e-3):
    '''
    Trains the model on all of the inputs and labels for one epoch.

    :param model: the initialized model to use for the forward pass and backward pass
    :param train_inputs: train inputs (all inputs to use for training), 
    shape (num_inputs, width, height, num_channels)
    :param train_labels: train labels (all labels to use for training), 
    shape (num_labels, num_classes)
    :return: Return the average accuracy across batches of the train inputs/labels
    '''
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    accuracy = []
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = model.loss(outputs, labels)
        accuracy.append(model.accuracy(outputs, labels))
        loss.backward()
        optimizer.step()
    return np.mean(accuracy)

def test(model, test_loader):
    """
    Tests the model on the test inputs and labels. You should NOT randomly 
    flip images or do any extra preprocessing.

    @param model: hi
    @param test_inputs: test data (all images to be tested), 
    shape (num_inputs, width, height, num_channels)
    :param test_labels: test labels (all corresponding labels),
    shape (num_labels, num_classes)
    :return: test accuracy - this should be the average accuracy across
    all batches
    """
    model.to("cpu")
    model.eval()
    accuracy = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs, labels
            outputs = model(inputs)
            accuracy.append(model.accuracy(outputs, labels))
    return np.mean(accuracy)

def interpret(model, interpret_loader, visualization_f, num_inputs=15, scuff_steps=50):
    '''
    ....

    :param model: the initialized model to use for the forward pass and backward pass
    :param train_inputs: train inputs (all inputs to use for training), 
    shape (num_inputs, width, height, num_channels)
    :param train_labels: train labels (all labels to use for training), 
    shape (num_labels, num_classes)
    :return: Return the average accuracy across batches of the train inputs/labels
    '''

    model.to("cpu")
    model.eval()
    scuffed_inputs = []

    def img_norm(a):
        return a / torch.sqrt(torch.clamp(torch.sum(torch.square(a)), min=1e-8))

    def img_dot(a, b):
        return torch.sum(torch.mul(a, b))

    for i, (inputs, labels) in enumerate(interpret_loader):
        if i == num_inputs:
            break
        label = torch.argmax(labels, dim=1).item()
        x_var = inputs.clone().detach().requires_grad_(True)
        scuffed_i = [inputs]
        for _ in range(scuff_steps):
            outputs = model(x_var, is_interpret=True)
            loss = model.loss(outputs, labels)
            if _ == 0:
                initial_pred = torch.nn.functional.softmax(outputs, dim=1)[0][label].clone().detach().tolist()
            model.zero_grad()
            x_var.retain_grad()
            loss.backward()
            grads = x_var.grad
            assert grads is not None
            grads = img_norm(grads)
            v = torch.ones_like(grads)
            v = img_norm(v)
            u_dot_v = img_dot(grads, v)
            perp = v - u_dot_v * grads
            perp = img_norm(perp)
            x_var = x_var + perp
            scuffed_i.append(x_var.clone().detach())
        final_outputs = torch.nn.functional.softmax(model(x_var, is_interpret=True), dim=1)
        final_pred = final_outputs[0][label].tolist()
        print(f"PRED: {initial_pred} -> {final_pred}")
        print(f"X: {inputs[0, :, 8, 8].tolist()} -> {x_var[0, :, 8, 8].tolist()}")
        scuffed_inputs.append(scuffed_i)

    visualization_f(scuffed_inputs)
    return scuffed_inputs

def load_or_train_model(ModelClass, train_loader, save_path, epochs=1, force_train=False):
    model = ModelClass()
    def do_train():
        for epoch in range(epochs):
            train_acc = train(model, train_loader)
            print(f"Epoch: {epoch}, Training Accuracy: {train_acc}")
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
    # Use the local filepaths when running on your local machine.
    AUTOGRADER_TRAIN_FILE = '../data/cifar10_train'
    AUTOGRADER_TEST_FILE = '../data/cifar10_test'

    train_loader = get_data(AUTOGRADER_TRAIN_FILE)
    test_loader = get_data(AUTOGRADER_TEST_FILE)
    interpret_loader = get_data(AUTOGRADER_TEST_FILE, batch_size=1)

    # Instantiate our model
    model = load_or_train_model(Cifar10Model, train_loader, '../models/cifar10.pt',
                                epochs=10, force_train=FORCE_TRAIN)

    test_acc = test(model, test_loader)
    print(f"Test Accuracy: {test_acc}")

    interpret(model, interpret_loader, save_tensor_gifs)


if __name__ == '__main__':
    main()