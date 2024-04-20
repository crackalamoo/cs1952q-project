import torch.nn as nn

# Activation function to be used across models
leaky_relu = nn.LeakyReLU(0.01)