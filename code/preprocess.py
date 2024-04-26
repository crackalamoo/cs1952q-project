import pickle

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torchvision
from torchvision import transforms
import shutil


def unpickle(file):
    """
    CIFAR data contains the files data_batch_1, data_batch_2, ..., 
    as well as test_batch. We have combined all train batches into one
    batch for you. Each of these files is a Python "pickled" 
    object produced with cPickle. The code below will open up a 
    "pickled" object (each file) and return a dictionary.

    :param file: the file to unpickle
    :return: dictionary of unpickled data
    """
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def pickle_mnist():
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
    trainset.data = trainset.data.tolist()
    trainset = {b'data': trainset.data, b'labels': trainset.targets}
    with open('../data/mnist_train', 'wb') as fo:
        pickle.dump(trainset, fo)
    
    testset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
    testset.data = testset.data.tolist()
    testset = {b'data': testset.data, b'labels': testset.targets}
    with open('../data/mnist_test', 'wb') as fo:
        pickle.dump(testset, fo)
    
    shutil.rmtree('./data')

def get_image_classifier_data(file_path, classes=None, num_channels=3, image_size=32, batch_size=64, shuffle=True):
    """
    Given a file path and two target classes, returns an array of 
    normalized inputs (images) and an array of labels. 
    You will want to first extract only the data that matches the 
    corresponding classes we want (there are 10 classes and we only want 2).
    You should make sure to normalize all inputs and also turn the labels
    into one hot vectors using tf.one_hot().
    Note that because you are using tf.one_hot() for your labels, your
    labels will be a Tensor, while your inputs will be a NumPy array. This 
    is fine because PyTorch works with NumPy arrays.
    :param file_path: file path for inputs and labels, something 
    like 'CIFAR_data_compressed/train'
    :param first_class:  an integer (0-9) representing the first target
    class in the CIFAR10 dataset, for a cat, this would be a 3
    :param second_class:  an integer (0-9) representing the second target
    class in the CIFAR10 dataset, for a dog, this would be a 5

    :return: normalized NumPy array of inputs and tensor of labels, where 
    inputs are of type np.float32 and has size (num_inputs, width, height, num_channels) and labels 
    has size (num_examples, num_classes)
    """
    unpickled_file = unpickle(file_path)
    inputs = unpickled_file[b'data']
    labels = unpickled_file[b'labels']

    inputs = np.asarray(inputs)
    labels = np.asarray(labels)

    if classes is not None:
        indexing = (labels == classes[0])
        for i in range(1, len(classes)):
            indexing |= (labels == classes[i])
        inputs = inputs[indexing]
        labels = labels[indexing]
        vfunc = np.vectorize(lambda l: classes.index(l))
        labels = vfunc(labels)

    inputs = torch.tensor(inputs, dtype=torch.float32)
    labels = torch.nn.functional.one_hot(torch.tensor(labels), num_classes=(10 if classes is None else len(classes)))
    idxs = torch.arange(0, inputs.size(0))

    # Reshape and transpose images to match PyTorch's convention (num_inputs, num_channels, width, height)
    inputs = inputs.view(-1, num_channels, image_size, image_size)

    # Normalize inputs
    inputs /= 255.0

    dataset = TensorDataset(inputs, labels, idxs)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return data_loader

if __name__ == '__main__':
    pickle_mnist()