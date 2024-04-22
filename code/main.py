from __future__ import absolute_import

import argparse
import numpy as np
import torch
from cifar10 import Cifar10Model, get_cifar10_data
from visuals import visualize_interpret_images

parser = argparse.ArgumentParser()
parser.add_argument("--force-train", action=argparse.BooleanOptionalAction)
parser.add_argument("--device")
parser.add_argument("--epochs")
args = parser.parse_args()

FORCE_TRAIN = args.force_train or False
EPOCHS = args.epochs or 10
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
    test_acc = np.mean(accuracy)
    print(f"Test Accuracy: {test_acc}")
    return test_acc

def interpret(model, interpret_loader, callback=None, num_inputs=5, scuff_steps=200, step_size=1):
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
    preds = []

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
        preds_i = []
        for i in range(scuff_steps):
            outputs = model(x_var, is_interpret=True)
            x_var_c = x_var.detach().clone()
            x_var_c = torch.clamp(x_var_c, 0, 1)
            outputs_c = model(x_var_c, is_interpret=True)
            pred = torch.nn.functional.softmax(outputs, dim=1)[0][label]
            pred_c = torch.nn.functional.softmax(outputs_c, dim=1)[0][label]
            if i == 0:
                initial_pred = pred_c
            preds_i.append(pred_c.detach().clone())
            model.zero_grad()
            x_var.retain_grad()
            pred.backward()
            grads = x_var.grad
            assert grads is not None
            grads = img_norm(grads)
            v = torch.ones_like(x_var)
            v = img_norm(v)
            u_dot_v = img_dot(grads, v)
            perp = v - u_dot_v * grads
            perp = img_norm(perp)
            x_var = x_var + step_size*perp
            x_var = x_var + step_size*grads
            x_var = x_var - step_size * 5e-3 * torch.where(x_var > 1, 1, 0)
            x_var = x_var + step_size * 5e-3 * torch.where(x_var < 0, 1, 0)
            x_var_c = torch.tensor(x_var.detach().numpy())
            x_var_c = torch.clamp(x_var_c, 0, 1)
            scuffed_i.append(x_var_c)
        final_outputs = torch.nn.functional.softmax(model(x_var_c.clone().detach(), is_interpret=True), dim=1)
        final_pred = final_outputs[0][label].tolist()
        print(f"PRED: {initial_pred} -> {final_pred}")
        print(f"X: {inputs[0, :, 8, 8].tolist()} -> {x_var[0, :, 8, 8].tolist()}")
        preds_i.append(final_pred)
        scuffed_inputs.append(scuffed_i)
        preds.append(preds_i)

    if callback is not None: callback(scuffed_inputs, preds)
    return scuffed_inputs, preds

def load_or_train_model(ModelClass, train_loader, save_path, epochs=10, force_train=False):
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
    train_loader, test_loader, interpret_loader = get_cifar10_data()

    model = load_or_train_model(Cifar10Model, train_loader, '../models/cifar10.pt',
                                epochs=EPOCHS, force_train=FORCE_TRAIN)

    test(model, test_loader)
    interpret(model, interpret_loader, callback=visualize_interpret_images)


if __name__ == '__main__':
    main()