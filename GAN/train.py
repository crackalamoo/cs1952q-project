import torch
from constants import *
from gan_core import GAN
from losses import d_acc_fake, d_acc_real, d_loss, g_acc, g_loss
from models import Discriminator, Generator
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from visualization import EpochVisualizer

# Instantiate models
dis_model = Discriminator(img_size)
gen_model = Generator(img_size, z_dim)

gan_model = GAN(dis_model=dis_model, gen_model=gen_model, z_dims=(None, z_dim))

# Setup the optimizer
d_optimizer = torch.optim.Adam(
    dis_model.parameters(), lr=1e-3, betas=(0.5, 0.999))
g_optimizer = torch.optim.Adam(
    gen_model.parameters(), lr=1e-3, betas=(0.5, 0.999))

# Load CIFAR10 data
transform = transforms.Compose([
    transforms.ToTensor(),                      # Convert images to PyTorch tensors
    # Normalize the tensors; CIFAR10 images are in RGB
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
cifar10_data = datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(cifar10_data, batch_size=50, shuffle=True)

# Prepare sample inputs for visualization
# 8 examples to generate images from for seeing the evolution
fixed_z = torch.randn(8, z_dim)
fixed_x, _ = next(iter(train_loader))
# Take the first 8 examples for consistency
print(fixed_x.shape)
fixed_x = fixed_x.view(fixed_x.size(0), -1)[:8]

visualizer = EpochVisualizer(gan_model, sample_inputs=(fixed_x, fixed_z))

# Training function


def train(gan_model, train_loader, visualizer, epochs, dis_steps, gen_steps):
    for epoch in range(epochs):
        for i, (real_images, _) in enumerate(train_loader):
            real_images = real_images.view(
                real_images.size(0), -1)  # Flatten the images
            batch_size = real_images.size(0)

            # Generate noise for generator input
            z = torch.randn(batch_size, z_dim)

            # Train Discriminator
            fake_images = gen_model(z).detach()
            real_pred = dis_model(real_images)
            fake_pred = dis_model(fake_images)
            d_loss_val = d_loss(fake_pred, real_pred)

            # Calculate discriminator accuracy on real and fake images
            d_acc_real_val = d_acc_real(real_pred)
            d_acc_fake_val = d_acc_fake(fake_pred)

            d_optimizer.zero_grad()
            d_loss_val.backward()
            d_optimizer.step()

            # Train Generator less frequently than the discriminator
            if (i+1) % dis_steps == 0:
                for _ in range(gen_steps):
                    fake_images = gen_model(z)
                    fake_pred = dis_model(fake_images)
                    g_loss_val = g_loss(fake_pred)
                    g_acc_val = g_acc(fake_pred)

                    g_optimizer.zero_grad()
                    g_loss_val.backward()
                    g_optimizer.step()

            if (i+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], '
                      f'D Loss: {d_loss_val.item():.4f}, G Loss: {g_loss_val.item():.4f}, '
                      f'D Acc Real: {d_acc_real_val.item():.4f}, D Acc Fake: {d_acc_fake_val.item():.4f}, G Acc: {g_acc_val.item():.4f}')

        # Call visualizer to process and save images at the end of each epoch
        visualizer.on_epoch_end(epoch)


# Call the train function with appropriate parameters
epochs = 10
dis_steps = 5
gen_steps = 5
train(gan_model, train_loader, visualizer, epochs, dis_steps, gen_steps)

# Optionally save a GIF of the training progress after all epochs
visualizer.save_gif(filename="cifar10_training_progress")
