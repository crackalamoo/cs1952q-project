import torch
import torchvision
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
        self.dropout_rate = 0.25

        self.dropout = torch.nn.Dropout(self.dropout_rate)

        self.res_fn1 = torch.nn.Conv2d(3, 16, 3, stride=1, padding='same')
        self.res_fn2 = torch.nn.Conv2d(16, 16, 3, stride=1, padding='same')
        self.res_skip1 = torch.nn.Conv2d(3, 16, 1, stride=1, padding='same')

        self.res_fn3 = torch.nn.Conv2d(16, 20, 3, stride=1, padding='same')
        self.res_fn4 = torch.nn.Conv2d(20, 20, 3, stride=1, padding='same')
        self.res_skip2 = torch.nn.Conv2d(16, 20, 1, stride=1, padding='same')

        self.res_fn5 = torch.nn.Conv2d(20, 32, 3, stride=1, padding='same')
        self.res_fn6 = torch.nn.Conv2d(32, 32, 3, stride=1, padding='same')
        self.res_skip3 = torch.nn.Conv2d(20, 32, 1, stride=1, padding='same')

        self.conv1 = torch.nn.Conv2d(32, 32, 3, stride=1)
        self.conv2 = torch.nn.Conv2d(32, 32, 3, stride=1)
        
        self.pool_fn = torch.nn.MaxPool2d(3, stride=2)

        self.batch_norm1 = torch.nn.BatchNorm2d(16)
        self.batch_norm2 = torch.nn.BatchNorm2d(16)
        self.batch_norm3 = torch.nn.BatchNorm2d(20)
        self.batch_norm4 = torch.nn.BatchNorm2d(20)
        self.batch_norm5 = torch.nn.BatchNorm2d(32)
        self.batch_norm6 = torch.nn.BatchNorm2d(32)
        self.batch_norm7 = torch.nn.BatchNorm2d(32)

        self.linear = torch.nn.Linear(32*5*5, self.hidden_layer1)
        self.linear2 = torch.nn.Linear(self.hidden_layer1, self.num_classes)
        self.linears = [self.linear, self.linear2]

        self.flatten = torch.nn.Flatten()

    def forward(self, inputs):
        # Remember that
        # shape of input = (num_inputs (or batch_size), in_height, in_width, in_channels)
        # shape of filter = (filter_height, filter_width, in_channels, out_channels)
        # shape of strides = (batch_stride, height_stride, width_stride, channels_stride)
        out = inputs

        # Residual block
        out = self.res_fn1(out)
        out = self.batch_norm1(out)
        out = torch.nn.functional.relu(out)
        out = self.res_fn2(out)
        out = torch.nn.functional.relu(out)
        out = self.res_skip1(inputs) + out
        out = self.batch_norm2(out)
        out = torch.nn.functional.relu(out)

        out = self.pool_fn(out)

        # Residual block
        pre_res = out
        out = self.res_fn3(out)
        out = self.batch_norm3(out)
        out = torch.nn.functional.relu(out)
        out = self.res_fn4(out)
        out = self.res_skip2(pre_res) + out
        out = self.batch_norm4(out)
        out = torch.nn.functional.relu(out)

        # Residual block
        pre_res = out
        out = self.res_fn5(out)
        out = self.batch_norm5(out)
        out = torch.nn.functional.relu(out)
        out = self.res_fn6(out)
        out = self.res_skip3(pre_res) + out
        out = self.batch_norm6(out)
        out = torch.nn.functional.relu(out)

        out = self.conv1(out)
        out = self.batch_norm7(out)
        out = torch.nn.functional.relu(out)
        out = self.conv2(out)
        out = torch.nn.functional.relu(out)

        out = self.pool_fn(out)

        logits = self.flatten(out)
        for i, linear in enumerate(self.linears):
            logits = linear(logits)
            if i < len(self.linears) - 1:
                logits = torch.nn.functional.relu(logits)
                logits = self.dropout(logits)

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
    test_loader = get_image_classifier_data(AUTOGRADER_TEST_FILE, shuffle=False)

    # train_loader is a dataloader. we want to add to it by adding a horizontal flip transform
    train_loader.dataset.transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ColorJitter(brightness=0.2, saturation=0.1),
    ])

    return train_loader, test_loader, None