from __future__ import absolute_import

import math
import os

import numpy as np
import tensorflow as tf
from convolution import conv2d
from matplotlib import pyplot as plt
from PIL import Image
from preprocess import get_data

# ensures that we run only on cpu
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


class Model(tf.keras.Model):
    def __init__(self):
        """
        This model class will contain the architecture for your CNN that 
        classifies images. We have left in variables in the constructor
        for you to fill out, but you are welcome to change them if you'd like.
        """
        super(Model, self).__init__()

        self.batch_size = 64
        self.num_classes = 2
        # Append losses to this list in training so you can visualize loss vs time in main
        self.loss_list = []

        # Initialize all hyperparameters
        self.learning_rate = 1e-3
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        self.hidden_layer1 = 2*2*20
        self.hidden_layer2 = 2*20
        self.dropout_rate = 0.25

        def conv_fn1(l): return tf.nn.conv2d(
            l, self.filter1, [1, 2, 2, 1], padding='SAME')

        def pool_fn1(l): return tf.nn.max_pool(l, 3, 2, padding='SAME')

        def conv_fn2(l): return tf.nn.conv2d(
            l, self.filter2, [1, 2, 2, 1], padding='SAME')

        def pool_fn2(l): return tf.nn.avg_pool(l, 2, 2, padding='SAME')

        def conv_fn3(l): return tf.nn.conv2d(
            l, self.filter3, [1, 1, 1, 1], padding='SAME')

        def pool_fn3(l): return l

        self.conv_fns = [conv_fn1, conv_fn2, conv_fn3]
        self.pool_fns = [pool_fn1, pool_fn2, pool_fn3]

        # Initialize all trainable parameters
        def make_variables(*dims, initializer=tf.random.truncated_normal):
            # *dims takes all unnamed variables and condenses to dims list
            return tf.Variable(initializer(dims, stddev=.1))

        self.filter1 = make_variables(5, 5, 3, 16)
        self.filter2 = make_variables(5, 5, 16, 20)
        self.filter3 = make_variables(3, 3, 20, 20)
        self.cnn_bias1 = make_variables(16)
        self.cnn_bias2 = make_variables(20)
        self.cnn_bias3 = make_variables(20)

        self.W1 = make_variables(2*2*20, self.hidden_layer1)
        self.b1 = make_variables(self.hidden_layer1)
        self.W2 = make_variables(self.hidden_layer1, self.hidden_layer2)
        self.b2 = make_variables(self.hidden_layer2)
        self.W3 = make_variables(self.hidden_layer2, self.num_classes)
        self.b3 = make_variables(self.num_classes)

        self.cnn_biases = [self.cnn_bias1, self.cnn_bias2, self.cnn_bias3]
        self.ws = [self.W1, self.W2, self.W3]
        self.bs = [self.b1, self.b2, self.b3]
        self.filters = [self.filter1, self.filter2, self.filter3]

        self.flatten = tf.keras.layers.Flatten()

    def call(self, inputs, is_testing=False, is_interpret=False):
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
        for i, conv_fn, pool_fn, bias in zip(range(len(self.cnn_biases)), self.conv_fns, self.pool_fns, self.cnn_biases):
            if i == len(self.cnn_biases)-1 and is_testing:
                out = conv2d(out, self.filters[-1], [1, 1, 1, 1], 'SAME')
            else:
                out = conv_fn(out)
            out = tf.nn.bias_add(out, bias)

            # Calculate mean and variance for batch normalization over the [batch, height, width] dimensions, keeping the channel dimension intact
            mean, var = tf.nn.moments(out, [0, 1, 2])
            out_channels = self.filters[i].shape[-1]
            beta = tf.Variable(tf.zeros([out_channels]), name='beta')
            scale = tf.Variable(tf.ones([out_channels]), name='scale')
            epsilon = 1e-3
            out = tf.nn.batch_normalization(
                out, mean, var, beta, scale, epsilon)

            out = tf.nn.relu(out)
            out = pool_fn(out)

        logits = self.flatten(out)
        for i, w, b in zip(range(len(self.ws)), self.ws, self.bs):
            if i != len(self.ws)-1 and not is_interpret:  # dropout
                w = tf.nn.dropout(w, self.dropout_rate)
            logits = tf.matmul(logits, w) + b

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

        loss = tf.nn.softmax_cross_entropy_with_logits(labels, logits)
        loss = tf.reduce_mean(loss)
        self.loss_list.append(loss)
        return loss

    def accuracy(self, logits, labels):
        """
        Calculates the model's prediction accuracy by comparing
        logits to correct labels – no need to modify this.

        :param logits: a matrix of size (num_inputs, self.num_classes); during training, this will be (batch_size, self.num_classes)
        containing the result of multiple convolution and feed forward layers
        :param labels: matrix of size (num_labels, self.num_classes) containing the answers, during training, this will be (batch_size, self.num_classes)

        NOTE: DO NOT EDIT

        :return: the accuracy of the model as a Tensor
        """
        correct_predictions = tf.equal(
            tf.argmax(logits, 1), tf.argmax(labels, 1))
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))


def train(model, train_inputs, train_labels):
    '''
    Trains the model on all of the inputs and labels for one epoch. You should shuffle your inputs 
    and labels - ensure that they are shuffled in the same order using tf.gather or zipping.
    To increase accuracy, you may want to use tf.image.random_flip_left_right on your
    inputs before doing the forward pass. You should batch your inputs.

    :param model: the initialized model to use for the forward pass and backward pass
    :param train_inputs: train inputs (all inputs to use for training), 
    shape (num_inputs, width, height, num_channels)
    :param train_labels: train labels (all labels to use for training), 
    shape (num_labels, num_classes)
    :return: Return the average accuracy across batches of the train inputs/labels
    '''

    # shuffle inputs and labels
    indices = tf.random.shuffle(range(len(train_inputs)))
    inputs = tf.gather(train_inputs, indices)
    labels = tf.gather(train_labels, indices)

    accuracy = []

    for b1 in range(model.batch_size, inputs.shape[0]+1, model.batch_size):
        b0 = b1 - model.batch_size
        x, y = inputs[b0:b1], labels[b0:b1]
        with tf.GradientTape() as tape:
            y_pred = model.call(x)
            loss = model.loss(y_pred, y)
            accuracy.append(model.accuracy(y_pred, y))

            grads = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(
                zip(grads, model.trainable_variables))

    return np.mean(accuracy)


def test(model, test_inputs, test_labels):
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
    accuracy = []

    for b1 in range(model.batch_size, test_inputs.shape[0]+1, model.batch_size):
        b0 = b1 - model.batch_size
        x, y = test_inputs[b0:b1], test_labels[b0:b1]

        y_pred = model.call(x)
        accuracy.append(model.accuracy(y_pred, y))

    return np.mean(accuracy)


def interpret(model, test_inputs, test_labels):
    '''
    ....

    :param model: the initialized model to use for the forward pass and backward pass
    :param train_inputs: train inputs (all inputs to use for training), 
    shape (num_inputs, width, height, num_channels)
    :param train_labels: train labels (all labels to use for training), 
    shape (num_labels, num_classes)
    :return: Return the average accuracy across batches of the train inputs/labels
    '''

    # shuffle inputs and labels
    indices = tf.random.shuffle(range(len(test_inputs)))
    inputs = tf.gather(test_inputs, indices)
    labels = tf.gather(test_labels, indices)

    scuffed_inputs = []

    def img_norm(a):
        return a / tf.math.sqrt(tf.math.maximum(
            tf.math.reduce_sum(tf.math.square(a)), 1e-8))

    def img_dot(a, b):
        return tf.math.reduce_sum(tf.math.multiply(a, b))

    for b in range(15):
        x, y = inputs[b:b+1], labels[b:b+1]
        x_var = tf.Variable(initial_value=x, trainable=True)
        for i in range(15):
            with tf.GradientTape() as tape:
                y_pred = tf.nn.softmax(model.call(x_var, is_interpret=True))[0][0]
                if i == 0:
                    initial_pred = tf.identity(y_pred)

                grads = tape.gradient(y_pred, x_var)
                assert grads is not None
            grads = img_norm(grads)
            v = tf.ones_like(grads)
            v = img_norm(v)
            u_dot_v = img_dot(grads, v)
            perp = v - u_dot_v * grads
            perp = img_norm(perp)
            x_var.assign_add(perp)
        final_pred = tf.nn.softmax(model.call(x_var, is_interpret=True))[0][0]
        print(f"PRED: {initial_pred} -> {final_pred}")
        print(f"X: {x[0][8][8][:]} -> {x_var[0][8][8][:]}")
        scuffed_inputs.append(x_var)

    return scuffed_inputs


def tensor_to_image(tensor):
    """
    Converts a TensorFlow tensor to a PIL Image and normalizes the data.
    """
    tensor = tensor * 255  # Assuming the tensor is in [0, 1]
    tensor = tf.cast(tensor, tf.uint8)
    return Image.fromarray(tensor.numpy())


def visualize(images, folder_path='../visualized_images'):
    """
    Saves a list of TensorFlow tensors as RGB images in the specified folder.

    Args:
    images (list of tf.Tensor): List of TensorFlow tensors to be converted to images.
    folder_path (str): Path to the folder where images will be saved.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    for idx, img_tensor in enumerate(images):
        # Check if the image tensor is a valid image format
        if len(img_tensor[0].shape) == 3 and img_tensor.shape[-1] == 3:
            img = tensor_to_image(img_tensor[0])
            img.save(os.path.join(folder_path, f'image_{idx+1}.png'))
        else:
            print(
                f"Skipping tensor {idx+1} as it does not conform to RGB image dimensions.")


def visualize_loss(losses):
    """
    Uses Matplotlib to visualize the losses of our model.
    :param losses: list of loss data stored from train. Can use the model's loss_list 
    field 

    NOTE: DO NOT EDIT

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

    NOTE: DO NOT EDIT

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

    LOCAL_TRAIN_FILE = ''
    LOCAL_TEST_FILE = ''

    # Instantiate our model
    model = Model()

    train_inputs, train_labels = get_data(
        AUTOGRADER_TRAIN_FILE, 3, 5)
    test_inputs, test_labels = get_data(
        AUTOGRADER_TEST_FILE, 3, 5)

    epochs = 1
    for e in range(epochs):
        x = train(model, train_inputs, train_labels)
        print(f"Epoch: {e}, Training Accuracy: {x}")

    y = test(model, test_inputs, test_labels)
    print(f"Test Accuracy: {y}")

    visualize(interpret(model, test_inputs, test_labels))


if __name__ == '__main__':
    main()
