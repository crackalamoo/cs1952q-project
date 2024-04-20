import io

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from constants import*


class EpochVisualizer:
    def __init__(self, model, sample_inputs):
        self.model = model
        self.sample_inputs = sample_inputs
        self.imgs = []

    def on_epoch_end(self, epoch):
        z_samp = torch.randn(8, z_dim)  # Re-sample z for each epoch
        x_fake = self.model.gen_model(z_samp).detach()
        x_real = self.sample_inputs[0]
        x_real = x_real.view(-1, channels, img_size, img_size)
        x_fake = x_fake.view(-1, channels, img_size, img_size)
        d_real = torch.sigmoid(self.model.dis_model(x_real)).detach()
        d_fake = torch.sigmoid(self.model.dis_model(x_fake)).detach()
        outputs = torch.cat([x_real[:4, :, :, :], x_fake[:4, :, :, :]], dim=0)
        labels = [
            f"D(true x) = {np.round(100 * d.item(), 0)}%" for d in d_real]
        labels += [f"D(fake x) = {np.round(100 * d.item(), 0)}%" for d in d_fake]

        self.add_to_imgs(
            outputs=outputs,
            labels=labels,
            epoch=epoch
        )

    def add_to_imgs(self, outputs, labels, epoch, nrows=1, ncols=8, figsize=(16, 2)):
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        for i, ax in enumerate(axs.flatten()):
            ax.clear()  # Clear previous plots
            img = outputs[i].squeeze().cpu().detach().numpy()
            ax.imshow(img, cmap='gray')
            ax.set_title(labels[i], fontsize=8)
            ax.axis('off')
        plt.tight_layout()
        self.imgs.append(self.fig2img(fig))
        plt.close(fig)

    @staticmethod
    def fig2img(fig):
        """
        Convert a Matplotlib figure to a PIL Image and return it
        """
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf)
        return img

    def save_gif(self, filename='mnist_recon', loop=0, duration=500):
        """
        Save the images as a GIF file
        """
        imgs = self.imgs  # This should be a list of PIL Image objects
        imgs[0].save(
            f'{filename}.gif', save_all=True, append_images=imgs[1:],
            loop=loop, duration=duration, format='GIF')
