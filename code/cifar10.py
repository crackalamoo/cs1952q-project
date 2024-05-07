import torch
from preprocess import get_image_classifier_data

class Cifar10Model(torch.nn.Module):
    def __init__(self):
        super(Cifar10Model, self).__init__()

        self.batch_size = 64
        self.num_classes = 10
        # Append losses to this list in training so you can visualize loss vs time in main
        self.loss_list = []

        # Initialize all hyperparameters
        self.hidden_layer1 = 16
        self.hidden_layer2 = 16
        self.dropout_rate = 0.25

        self.res_fn1 = torch.nn.Conv2d(3, 16, 3, stride=1, padding='same')
        self.res_fn2 = torch.nn.Conv2d(16, 16, 3, stride=1, padding='same')
        self.res_skip1 = torch.nn.Conv2d(3, 16, 1, stride=1, padding='same')

        self.res_fn3 = torch.nn.Conv2d(16, 20, 3, stride=1, padding='same')
        self.res_fn4 = torch.nn.Conv2d(20, 20, 3, stride=1, padding='same')
        self.res_skip2 = torch.nn.Conv2d(16, 20, 1, stride=1, padding='same')
        
        self.pool_fn = torch.nn.MaxPool2d(3, stride=2)

        self.batch_norm1 = torch.nn.BatchNorm2d(16)
        self.batch_norm2 = torch.nn.BatchNorm2d(20)
        self.batch_norm3 = torch.nn.BatchNorm2d(20)

        # self.conv_fns = [self.conv_fn1]
        # self.pool_fns = [self.pool_fn1]
        # self.batch_fns = [self.batch_norm1]
        # self.dropout = torch.nn.Dropout(self.dropout_rate)
        self.linear = torch.nn.Linear(20*7*7, self.hidden_layer1)
        self.hidden1 = torch.nn.Linear(self.hidden_layer1, self.hidden_layer1)
        self.hidden2 = torch.nn.Linear(self.hidden_layer1, self.num_classes)
        self.linears = [self.linear, self.hidden2]

        self.flatten = torch.nn.Flatten()

    def forward(self, inputs):
        # Remember that
        # shape of input = (num_inputs (or batch_size), in_height, in_width, in_channels)
        # shape of filter = (filter_height, filter_width, in_channels, out_channels)
        # shape of strides = (batch_stride, height_stride, width_stride, channels_stride)
        out = inputs

        # Residual block
        out = self.res_fn1(out)
        out = torch.nn.functional.relu(out)
        out = self.res_fn2(out)
        out = torch.nn.functional.relu(out)
        out = self.batch_norm1(out)
        out = self.res_skip1(inputs) + out
        out = torch.nn.functional.relu(out)

        out = self.pool_fn(out)


        # Residual block
        pre_res = out
        out = self.res_fn3(out)
        out = torch.nn.functional.relu(out)
        out = self.res_fn4(out)
        out = self.batch_norm2(out)
        out = self.res_skip2(pre_res) + out
        out = torch.nn.functional.relu(out)

        out = self.pool_fn(out)

        logits = self.flatten(out)
        for linear in self.linears:
            logits = linear(logits)

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

def get_cifar10_data():
    AUTOGRADER_TRAIN_FILE = '../data/cifar10_train'
    AUTOGRADER_TEST_FILE = '../data/cifar10_test'

    train_loader = get_image_classifier_data(AUTOGRADER_TRAIN_FILE)
    test_loader = get_image_classifier_data(AUTOGRADER_TEST_FILE)

    return train_loader, test_loader, None