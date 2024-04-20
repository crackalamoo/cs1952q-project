import torch
import torch.nn as nn

class GAN(nn.Module):
    '''
    Core class to administrate both the generator and discriminator
    '''

    def __init__(self, dis_model, gen_model, z_dims, z_sampler=None):
        '''
        self.gen_model = generator model;           z_like -> x_like
        self.dis_model = discriminator model;       x_like -> probability
        self.z_sampler = sampling strategy for z;   z_dims -> z
        self.z_dims    = dimensionality of generator input
        '''
        super(GAN, self).__init__()
        self.z_dims = z_dims
        self.z_sampler = z_sampler if z_sampler is not None else self.default_z_sampler
        self.gen_model = gen_model
        self.dis_model = dis_model

    def default_z_sampler(self, num_samples):
        '''
        Default z sampler generates normal distributed vectors
        Shape of z: [num_samples, *self.z_dims[1:]]
        '''
        return torch.randn((num_samples, *self.z_dims[1:]))

    def sample_z(self, num_samples):
        '''generates a z based on the z sampler'''
        return self.z_sampler(num_samples)

    def discriminate(self, x):
        '''predict whether input input is a real entry from the true dataset'''
        return self.dis_model(x)

    def generate(self, z):
        '''generates an output based on a specific z realization'''
        return self.gen_model(z)

    def forward(self, num_samples):
        '''
        Forward pass links generator and discriminator:
         - Generate a z sample
         - Generate an x-like image from z
         - Predict whether x-like is real
        '''
        z_samp = self.sample_z(num_samples)  
        g_samp = self.generate(z_samp)       
        d_samp = self.discriminate(g_samp)

        return d_samp