import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import math

def tensor_to_image(tensor):
    """
    Converts a PyTorch tensor to a PIL Image and normalizes the data.
    """
    tensor = tensor * 255  # Assuming the tensor is in [0, 1]
    tensor = tensor.to(dtype=torch.uint8)
    tensor = torch.permute(tensor, (1, 2, 0))
    return Image.fromarray(tensor.numpy())

def save_tensor_gifs(images, folder_path='../visualized_images'):
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