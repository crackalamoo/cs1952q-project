import torch
from preprocess import get_image_classifier_data

class MNISTModel(torch.nn.Module):
    def __init__(self, extras=None):
        super(MNISTModel, self).__init__()

        self.batch_size = 64
        self.num_classes = 10
        # Append losses to this list in training so you can visualize loss vs time in main
        self.loss_list = []

        # Initialize all hyperparameters
        self.hidden_layer1 = 16
        self.hidden_layer2 = 16
        self.dropout_rate = 0.25

        self.conv_fn1 = torch.nn.Conv2d(1, 16, 5, stride=2, padding=2)
        self.conv_fn2 = torch.nn.Conv2d(16, 20, 5, stride=2, padding=2)
        self.conv_fn3 = torch.nn.Conv2d(20, 20, 3, stride=1, padding='same')
        pool_fn1 = torch.nn.MaxPool2d(3, stride=2)
        self.batch_norm1 = torch.nn.BatchNorm2d(16)
        self.batch_norm2 = torch.nn.BatchNorm2d(20)
        self.batch_norm3 = torch.nn.BatchNorm2d(20)

        self.conv_fns = [self.conv_fn1]
        self.pool_fns = [pool_fn1]
        self.dropout = torch.nn.Dropout(self.dropout_rate)
        self.linear = torch.nn.Linear(16*6*6, self.hidden_layer1)
        self.hidden1 = torch.nn.Linear(self.hidden_layer1, self.hidden_layer1)
        self.hidden2 = torch.nn.Linear(self.hidden_layer1, self.num_classes)
        self.linears = [self.linear, self.hidden2]

        self.flatten = torch.nn.Flatten()

    def forward(self, inputs):
        out = inputs
        for i, conv_fn, pool_fn in zip(range(len(self.conv_fns)), self.conv_fns, self.pool_fns):
            out = conv_fn(out)

            # Calculate mean and variance for batch normalization over the [batch, height, width] dimensions, keeping the channel dimension intact
            # out = batch_fn(out)

            out = torch.nn.functional.relu(out)
            out = pool_fn(out)

        logits = self.flatten(out)
        for i, linear in enumerate(self.linears):
            logits = linear(logits)
            if i < len(self.linears) - 1:
                logits = torch.nn.functional.relu(logits)

        return logits

    def loss(self, logits, labels):
        loss = torch.nn.CrossEntropyLoss()(logits, torch.argmax(labels, dim=1))
        self.loss_list.append(loss)
        return loss
    
    def batch_losses(self, logits, labels):
        loss = torch.nn.CrossEntropyLoss(reduction='none')(logits, torch.argmax(labels, dim=1))
        return loss

    def accuracy(self, logits, labels):
        predicted = torch.argmax(logits, dim=1)
        correct = torch.argmax(labels, dim=1)
        correct_predictions = (predicted == correct).sum().item()
        total = labels.size(0)
        return correct_predictions / total

def get_mnist_data():
    AUTOGRADER_TRAIN_FILE = '../data/mnist_train'
    AUTOGRADER_TEST_FILE = '../data/mnist_test'

    get_mnist = lambda f: get_image_classifier_data(f, image_size=28, num_channels=1)
    train_loader = get_mnist(AUTOGRADER_TRAIN_FILE)
    test_loader = get_mnist(AUTOGRADER_TEST_FILE)

    return train_loader, test_loader, None