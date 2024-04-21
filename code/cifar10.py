import torch

class Cifar10Model(torch.nn.Module):
    def __init__(self):
        """
        This model class will contain the architecture for your CNN that 
        classifies images. We have left in variables in the constructor
        for you to fill out, but you are welcome to change them if you'd like.
        """
        super(Cifar10Model, self).__init__()

        self.batch_size = 64
        self.num_classes = 10
        # Append losses to this list in training so you can visualize loss vs time in main
        self.loss_list = []

        # Initialize all hyperparameters
        self.hidden_layer1 = 2*2*20
        self.hidden_layer2 = 2*20
        self.dropout_rate = 0.25

        self.conv_fn1 = torch.nn.Conv2d(3, 16, 5, stride=2, padding=2)
        self.conv_fn2 = torch.nn.Conv2d(16, 20, 5, stride=2, padding=2)
        self.conv_fn3 = torch.nn.Conv2d(20, 20, 3, stride=1, padding='same')
        pool_fn1 = torch.nn.MaxPool2d(3, stride=2, padding=1)
        pool_fn2 = torch.nn.AvgPool2d(2, stride=2)
        pool_fn3 = lambda l: l
        self.batch_norm1 = torch.nn.BatchNorm2d(16)
        self.batch_norm2 = torch.nn.BatchNorm2d(20)
        self.batch_norm3 = torch.nn.BatchNorm2d(20)

        self.conv_fns = [self.conv_fn1, self.conv_fn2, self.conv_fn3]
        self.pool_fns = [pool_fn1, pool_fn2, pool_fn3]
        self.batch_fns = [self.batch_norm1, self.batch_norm2, self.batch_norm3]
        self.dropout = torch.nn.Dropout(self.dropout_rate)
        self.linear = torch.nn.Linear(20*2*2, self.hidden_layer1)
        self.hidden1 = torch.nn.Linear(self.hidden_layer1, self.hidden_layer2)
        self.hidden2 = torch.nn.Linear(self.hidden_layer2, self.num_classes)
        self.linears = [self.linear, self.hidden1, self.hidden2]

        self.flatten = torch.nn.Flatten()

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