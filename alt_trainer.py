import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from utils.importer import import_model_from_config
from utils.plots import plot_grid
from models.separation_loss import WeightSeparationLoss

class ConvolutionalAutoencoder(nn.Module):
    def __init__(self, input_channels, image_hw, channels, hidden, num_encoders, norm_type):
        super(ConvolutionalAutoencoder, self).__init__()

        # Assuming 'channels' is a list like [24, 48, 96, 144] and defines the encoder architecture
        # Encoder layers
        encoder_layers = []
        current_channels = input_channels
        for out_channels in channels:
            encoder_layers.append(nn.Conv2d(current_channels, out_channels, kernel_size=3, stride=2, padding=1))
            if norm_type == 'batch_norm':
                encoder_layers.append(nn.BatchNorm2d(out_channels))
            elif norm_type == 'group_norm':
                encoder_layers.append(nn.GroupNorm(num_groups=2, num_channels=out_channels))
            encoder_layers.append(nn.ReLU(inplace=True))
            current_channels = out_channels

        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder layers - reverse of encoder layers
        decoder_layers = []
        channels.reverse()  # Reverse the channels to build the decoder
        for out_channels in channels[1:]:  # Skip the first one because it is handled separately
            decoder_layers.append(nn.ConvTranspose2d(current_channels, out_channels, kernel_size=3, stride=2, padding=1,
                                                     output_padding=1))
            if norm_type == 'batch_norm':
                decoder_layers.append(nn.BatchNorm2d(out_channels))
            elif norm_type == 'group_norm':
                decoder_layers.append(nn.GroupNorm(num_groups=2, num_channels=out_channels))
            decoder_layers.append(nn.ReLU(inplace=True))
            current_channels = out_channels

        # Final layer to match input channels
        decoder_layers.append(
            nn.ConvTranspose2d(current_channels, input_channels, kernel_size=3, stride=2, padding=1, output_padding=1))
        decoder_layers.append(nn.Sigmoid())  # Assuming we want the output in [0,1] for an image

        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# Define model parameters directly
input_channels = 1
image_hw = 64
channels = [24, 48, 96, 144]
hidden = 96
use_weight_norm = True
num_encoders = 2
norm_type = 'group_norm'

# Instantiate the model
model = ConvolutionalAutoencoder(input_channels, image_hw, channels, hidden, num_encoders, norm_type)

# Define hyperparameters directly
lr = 1e-3
lr_step_size = 50
weight_decay = 1e-5
z_decay = 1e-2
sep_lr = 5e-1
zero_lr = 1e-2
max_epochs = 300
batch_size = 512
plot_step = 50
save_plots = True
plot_dir = "./plots/"

if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

# Define loss functions
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=0.1)
separation_loss = WeightSeparationLoss(num_encoders, 'L1')

import torch
from torch.utils.data import Dataset
import pickle

class CustomDataset(Dataset):
    def __init__(self, pickle_file_path, transform=None):
        """
        Args:
            pickle_file_path (string): Path to the pickle file with the dataset.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        with open(pickle_file_path, 'rb') as f:
            og_data = torch.tensor(pickle.load(f))

        self.data = []
        for i in range(og_data.shape[0]):
            self.data.append(og_data[i])

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample


# Training loop
for epoch in range(max_epochs):
    model.train()
    # Replace 'dataset' with your actual dataset

    dataset = CustomDataset('data/single_channel_nonlinear_mixing_tri_circ.pickle')

    for batch_idx, (x, c, t) in enumerate(DataLoader(dataset, batch_size=batch_size, shuffle=True)):
        x_pred, z = model(x)
        recon_loss = criterion(x_pred, x)
        loss = recon_loss + z_decay * sum(torch.mean(z_i ** 2) for z_i in z)  # Incorporating z_decay

        sep_loss_value = separation_loss(model.decoder) if 'sep_loss' in globals() else 0
        loss += sep_loss_value * sep_lr

        zero_recon_loss = 0
        if 'zero_loss' in globals():
            z_zeros = [torch.zeros_like(z_i) for z_i in z]
            x_pred_zeros = model.decode(z_zeros)
            zero_recon_loss = criterion(x_pred_zeros, torch.zeros_like(x_pred_zeros))
            loss += zero_recon_loss * zero_lr

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    scheduler.step()

    # Validation step can be added similarly to the training step
