import pickle

import numpy as np
import torch
from torch import optim, nn
from torch.optim.lr_scheduler import StepLR

from models.cnn_multi_enc_ae_2d import ConvolutionalAutoencoder
from torch.utils.data import Dataset, DataLoader

from models.separation_loss import WeightSeparationLoss


class CircleTriangleDataset(Dataset):
    def __init__(self, transform=None):
        """
        Args:
            pickle_file_path (string): Path to the pickle file with the dataset.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        with open('data/single_channel_nonlinear_mixing_tri_circ.pickle', 'rb') as f:
            print('before extraction')
            self.data = pickle.load(f, encoding='latin1')
            print('after extraction')


        #self.data = []
        #for i in range(og_data.shape[0]):
        #    self.data.append(og_data[i])
#
        #self.transform = transform

    def __len__(self):
        return len(self.data)


    def min_max(self, x):
        return (x - np.min(x)) / (np.max(x) - np.min(x))


    def __getitem__(self, idx):
        x, c, t = self.data[idx]
        x, c, t = self.min_max(x), self.min_max(c), self.min_max(t)
        x = torch.tensor(x, dtype=torch.float32)
        c = torch.tensor(c, dtype=torch.float32)
        t = torch.tensor(t, dtype=torch.float32)

        return x.permute(2, 0, 1), c.permute(2, 0, 1), t.permute(2, 0, 1)

def get_model(input_channels=1, image_hw=64, channels=[32, 64, 128], hidden=512,
                 num_encoders=2, norm_type='none',
                 use_weight_norm=True):
    return ConvolutionalAutoencoder(input_channels, image_hw, channels, hidden, num_encoders, norm_type, use_weight_norm)


def get_dataloader(ct=True, batch_size=32, num_workers=12):
    if ct:
        return DataLoader(CircleTriangleDataset(), batch_size=batch_size, shuffle=True, num_workers=num_workers)


def train(model: ConvolutionalAutoencoder, dataloader_train, sep_norm='L1', sep_lr=0.5, zero_lr=0.01, lr=1e-3, lr_step_size=50, weight_decay=1e-5, z_decay=1e-2, max_epochs=300, name=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=lr_step_size, gamma=0.1)

    recon_loss = nn.MSELoss()
    sep_loss = WeightSeparationLoss(model.num_encoders, sep_norm)

    for epoch in range(max_epochs):
        model.train()
        train_loss = 0.0

        for data in dataloader_train:
            x = data[0].to(device)

            optimizer.zero_grad()

            # Forward pass
            x_pred, z = model(x)

            # Reconstruction loss
            loss = recon_loss(x_pred, x)

            # Add L2 regularisation penalising complex solutions
            for z_i in z:
                loss += loss + z_decay * torch.mean(z_i**2)

            # Add separation (i.e., sparse mixing) loss
            loss += sep_loss(model.decoder)*sep_lr

            # Add zero loss
            z_zeros = [torch.zeros(x.shape[0], model.hidden // model.num_encoders, z[0].shape[-1], z[0].shape[-1]).to(
                device) for _ in range(model.num_encoders)]
            x_pred_zeros = model.decode(z_zeros, True)
            zero_recon_loss = recon_loss(x_pred_zeros, torch.zeros_like(x_pred_zeros))
            loss += zero_recon_loss * zero_lr

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0)

        scheduler.step()

        train_loss /= len(dataloader_train.dataset)
        print(f'Epoch {epoch + 1}/{max_epochs}, Train Loss: {train_loss:.4f}')

