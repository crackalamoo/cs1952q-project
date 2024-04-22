import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
import math

def tensor_to_numpy_image(tensor):
    """
    Converts a PyTorch tensor to a NumPy Image and normalizes the data.
    """
    tensor = tensor * 255  # Assuming the tensor is in [0, 1]
    tensor = tensor.to(dtype=torch.uint8)
    tensor = torch.permute(tensor, (1, 2, 0))
    res = tensor.numpy()
    return res
def tensor_to_image(tensor):
    """
    Converts a PyTorch tensor to a PIL Image and normalizes the data.
    """
    return Image.fromarray(tensor_to_numpy_image(tensor))

def visualize_interpret_images(images, preds, folder_path='../visualized_images'):
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
    
    preds = np.array(preds)
    pred_len = preds.shape[1]
    # best_pred_idx = np.argmax(preds[:,0])
    best_pred_idx = np.random.randint(0, preds.shape[0])
    best_pred = preds[best_pred_idx,:]
    plot_img_idxs = [0, pred_len//4, pred_len*2//4, pred_len*3//4, pred_len-1]
    best_pred_imgs = list(map(lambda i: tensor_to_numpy_image(i[0]), images[best_pred_idx]))
    y_img_size = np.max(best_pred) - np.min(best_pred)
    
    
    plt.plot(np.arange(len(best_pred)), best_pred)
    plt.scatter(plot_img_idxs, best_pred[plot_img_idxs])
    for idx in plot_img_idxs:
        img = best_pred_imgs[idx]
        x = np.clip(idx, pred_len*0.05, pred_len*0.95)
        y = best_pred[idx]
        if y > 0.5*y_img_size+np.min(best_pred):
            y -= 0.1*y_img_size
        else:
            y += 0.1*y_img_size
        ab = AnnotationBbox(OffsetImage(img, zoom=2), (x,y), frameon=False)
        plt.gca().add_artist(ab)
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