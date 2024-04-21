from __future__ import absolute_import

import math
import os

import numpy as np
import torch
from convolution import conv2d
from matplotlib import pyplot as plt
from PIL import Image
from preprocess import get_data

# ensures that we run only on cpu
device = "cpu"


class Model(torch.nn.Module):
    def __init__(self):
        """
        This model class will contain the architecture for your CNN that 
        classifies images. We have left in variables in the constructor
        for you to fill out, but you are welcome to change them if you'd like.
        """
        super(Model, self).__init__()

        self.batch_size = 64
        self.num_classes = 10
        # Append losses to this list in training so you can visualize loss vs time in main
        self.loss_list = []

        # Initialize all hyperparameters
        self.learning_rate = 1e-3
        self.hidden_layer1 = 2*2*20
        self.hidden_layer2 = 2*20
        self.dropout_rate = 0.25

        conv_fn1 = torch.nn.Conv2d(3, 16, 5, stride=2, padding=2)
        conv_fn2 = torch.nn.Conv2d(16, 20, 5, stride=2, padding=2)
        conv_fn3 = torch.nn.Conv2d(20, 20, 3, stride=1, padding='same')
        pool_fn1 = torch.nn.MaxPool2d(3, stride=2, padding=1)
        pool_fn2 = torch.nn.AvgPool2d(2, stride=2)
        pool_fn3 = lambda l: l
        batch_norm1 = torch.nn.BatchNorm2d(16)
        batch_norm2 = torch.nn.BatchNorm2d(20)
        batch_norm3 = torch.nn.BatchNorm2d(20)

        self.conv_fns = [conv_fn1, conv_fn2, conv_fn3]
        self.pool_fns = [pool_fn1, pool_fn2, pool_fn3]
        self.batch_fns = [batch_norm1, batch_norm2, batch_norm3]
        self.dropout = torch.nn.Dropout(self.dropout_rate)
        self.linear = torch.nn.Linear(20*2*2, self.hidden_layer1)
        self.hidden1 = torch.nn.Linear(self.hidden_layer1, self.hidden_layer2)
        self.hidden2 = torch.nn.Linear(self.hidden_layer2, self.num_classes)
        self.linears = [self.linear, self.hidden1, self.hidden2]

        self.flatten = torch.nn.Flatten()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, inputs, is_testing=False, is_interpret=False):
        """
        Runs a forward pass on an input batch of images.

        :param inputs: images, shape of (num_inputs, 32, 32, 3); during training, the shape is (batch_size, 32, 32, 3)
        :param is_testing: a boolean that should be set to True only when you're doing Part 2 of the assignment and this function is being called during testing
        :return: logits - a matrix of shape (num_inputs, num_classes); during training, it would be (batch_size, 2)
        """
        # Remember that
        # shape of input = (num_inputs (or batch_size), in_height, in_width, in_channels)
        # shape of filter = (filter_height, filter_width, in_channels, out_channels)
        # shape of strides = (batch_stride, height_stride, width_stride, channels_stride)
        out = inputs
        for i, conv_fn, batch_fn, pool_fn in zip(range(len(self.conv_fns)), self.conv_fns, self.batch_fns, self.pool_fns):
            out = conv_fn(out)

            # Calculate mean and variance for batch normalization over the [batch, height, width] dimensions, keeping the channel dimension intact
            out = batch_fn(out)

            out = torch.nn.functional.relu(out)
            out = pool_fn(out)

        logits = self.flatten(out)
        for i, linear in enumerate(self.linears):
            if i != len(self.linears)-1 and not is_interpret:  # dropout
                logits = self.dropout(logits)
            logits = linear(logits)

        return logits

    def loss(self, logits, labels):
        """
        Calculates the model cross-entropy loss after one forward pass.
        Softmax is applied in this function.

        :param logits: during training, a matrix of shape (batch_size, self.num_classes) 
        containing the result of multiple convolution and feed forward layers
        :param labels: during training, matrix of shape (batch_size, self.num_classes) containing the train labels
        :return: the loss of the model as a Tensor
        """

        loss = torch.nn.CrossEntropyLoss()(logits, torch.argmax(labels, dim=1))
        self.loss_list.append(loss)
        return loss

    def accuracy(self, logits, labels):
        """
        Calculates the model's prediction accuracy by comparing
        logits to correct labels – no need to modify this.

        :param logits: a matrix of size (num_inputs, self.num_classes); during training, this will be (batch_size, self.num_classes)
        containing the result of multiple convolution and feed forward layers
        :param labels: matrix of size (num_labels, self.num_classes) containing the answers, during training, this will be (batch_size, self.num_classes)

        :return: the accuracy of the model as a Tensor
        """
        predicted = torch.argmax(logits, dim=1)
        correct = torch.argmax(labels, dim=1)
        correct_predictions = (predicted == correct).sum().item()
        total = labels.size(0)
        return correct_predictions / total


def train(model, train_loader):
    '''
    Trains the model on all of the inputs and labels for one epoch.

    :param model: the initialized model to use for the forward pass and backward pass
    :param train_inputs: train inputs (all inputs to use for training), 
    shape (num_inputs, width, height, num_channels)
    :param train_labels: train labels (all labels to use for training), 
    shape (num_labels, num_classes)
    :return: Return the average accuracy across batches of the train inputs/labels
    '''
    model.train()
    accuracy = []
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        model.optimizer.zero_grad()
        outputs = model(inputs)
        loss = model.loss(outputs, labels)
        accuracy.append(model.accuracy(outputs, labels))
        loss.backward()
        model.optimizer.step()
    return np.mean(accuracy)


def test(model, test_loader):
    """
    Tests the model on the test inputs and labels. You should NOT randomly 
    flip images or do any extra preprocessing.

    :param test_inputs: test data (all images to be tested), 
    shape (num_inputs, width, height, num_channels)
    :param test_labels: test labels (all corresponding labels),
    shape (num_labels, num_classes)
    :return: test accuracy - this should be the average accuracy across
    all batches
    """
    model.eval()
    accuracy = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            accuracy.append(model.accuracy(outputs, labels))
    return np.mean(accuracy)


def interpret(model, interpret_loader, num_images=15, scuff_steps=50):
    '''
    ....

    :param model: the initialized model to use for the forward pass and backward pass
    :param train_inputs: train inputs (all inputs to use for training), 
    shape (num_inputs, width, height, num_channels)
    :param train_labels: train labels (all labels to use for training), 
    shape (num_labels, num_classes)
    :return: Return the average accuracy across batches of the train inputs/labels
    '''

    model.eval()
    scuffed_inputs = []

    def img_norm(a):
        return a / torch.sqrt(torch.clamp(torch.sum(torch.square(a)), min=1e-8))

    def img_dot(a, b):
        return torch.sum(torch.mul(a, b))

    for i, (inputs, labels) in enumerate(interpret_loader):
        if i == num_images:
            break
        inputs, labels = inputs.to(device), labels.to(device)
        label = torch.argmax(labels, dim=1).item()
        x_var = torch.tensor(inputs.clone().detach(), device=device, requires_grad=True)
        scuffed_i = [inputs]
        for _ in range(scuff_steps):
            outputs = torch.nn.functional.softmax(model(x_var, is_interpret=True), dim=1)[0][label]
            if _ == 0:
                initial_pred = outputs.clone().detach().tolist()
            model.zero_grad()
            x_var.retain_grad()
            outputs.backward()
            x_var.retain_grad()
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

    return scuffed_inputs


def tensor_to_image(tensor):
    """
    Converts a PyTorch tensor to a PIL Image and normalizes the data.
    """
    tensor = tensor * 255  # Assuming the tensor is in [0, 1]
    tensor = tensor.to(dtype=torch.uint8)
    tensor = torch.permute(tensor, (1, 2, 0))
    return Image.fromarray(tensor.numpy())


def visualize_interpret(images, folder_path='../visualized_images'):
    """
    Saves a list of PyTorch tensors as RGB images in the specified folder.

    Args:
    images (list of List[torch.Tensor]): List of lists of PyTorch tensors to be converted to animations.
    folder_path (str): Path to the folder where images will be saved.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    for idx, img_tensors in enumerate(images):
        # Check if the image tensor is a valid image format
        imgs = [tensor_to_image(img_tensor[0]) for img_tensor in img_tensors]
        with open(os.path.join(folder_path, f'image_{idx+1}.gif'), 'wb') as f:
            imgs[0].save(f, save_all=True, append_images=imgs[1:], duration=100, loop=0)


def visualize_loss(losses):
    """
    Uses Matplotlib to visualize the losses of our model.
    :param losses: list of loss data stored from train. Can use the model's loss_list 
    field 

    :return: doesn't return anything, a plot should pop-up 
    """
    x = [i for i in range(len(losses))]
    plt.plot(x, losses)
    plt.title('Loss per batch')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.show()


def visualize_results(image_inputs, probabilities, image_labels, first_label, second_label):
    """
    Uses Matplotlib to visualize the correct and incorrect results of our model.
    :param image_inputs: image data from get_data(), limited to 50 images, shape (50, 32, 32, 3)
    :param probabilities: the output of model.call(), shape (50, num_classes)
    :param image_labels: the labels from get_data(), shape (50, num_classes)
    :param first_label: the name of the first class, "cat"
    :param second_label: the name of the second class, "dog"

    :return: doesn't return anything, two plots should pop-up, one for correct results,
    one for incorrect results
    """
    # Helper function to plot images into 10 columns
    def plotter(image_indices, label):
        nc = 10
        nr = math.ceil(len(image_indices) / 10)
        fig = plt.figure()
        fig.suptitle(
            "{} Examples\nPL = Predicted Label\nAL = Actual Label".format(label))
        for i in range(len(image_indices)):
            ind = image_indices[i]
            ax = fig.add_subplot(nr, nc, i+1)
            ax.imshow(image_inputs[ind], cmap="Greys")
            pl = first_label if predicted_labels[ind] == 0.0 else second_label
            al = first_label if np.argmax(
                image_labels[ind], axis=0) == 0 else second_label
            ax.set(title="PL: {}\nAL: {}".format(pl, al))
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.tick_params(axis='both', which='both', length=0)

    predicted_labels = np.argmax(probabilities, axis=1)
    num_images = image_inputs.shape[0]

    # Separate correct and incorrect images
    correct = []
    incorrect = []
    for i in range(num_images):
        if predicted_labels[i] == np.argmax(image_labels[i], axis=0):
            correct.append(i)
        else:
            incorrect.append(i)

    plotter(correct, 'Correct')
    plotter(incorrect, 'Incorrect')
    plt.show()


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
    #       Use the local filepaths when running on your local machine.
    AUTOGRADER_TRAIN_FILE = '../data/train'
    AUTOGRADER_TEST_FILE = '../data/test'

    # Instantiate our model
    model = Model()

    train_loader = get_data(AUTOGRADER_TRAIN_FILE)
    test_loader = get_data(AUTOGRADER_TEST_FILE)
    interpret_loader = get_data(AUTOGRADER_TEST_FILE, batch_size=1)

    epochs = 1
    for epoch in range(epochs):
        train_acc = train(model, train_loader)
        print(f"Epoch: {epoch}, Training Accuracy: {train_acc}")

    test_acc = test(model, test_loader)
    print(f"Test Accuracy: {test_acc}")

    scuffed_inputs = interpret(model, interpret_loader)
    visualize_interpret(scuffed_inputs)


if __name__ == '__main__':
    main()
