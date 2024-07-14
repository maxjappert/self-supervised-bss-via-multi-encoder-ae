import datetime
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torchaudio
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision import models
from torchvision.models import ResNet18_Weights

from models.cnn_ae_2d_spectrograms import *

from evaluation_metric_functions import compute_spectral_metrics
from functions import metric_index_mapping, save_spectrogram_to_file

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PriorDataset(Dataset):
    def __init__(self, split, debug=False):
        self.data_point_names = []
        self.master_path = os.path.join('data', 'musdb_18_prior', split)

        for data_folder in os.listdir(self.master_path):
            if random.random() < 0.9 and debug:
                continue

            for stem_idx in range(1, 5):
                self.data_point_names.append(os.path.join(data_folder, f'stem{stem_idx}'))

        self.transforms = transforms.Compose([
            transforms.ToTensor(),  # Convert numpy array to tensor
            transforms.Resize((1024, 384))
            # TODO: Try normalisation
            #transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize to mean=0.5, std=0.5
            #transforms.Lambda(lambda x: torch.log(x + 1e-9)),  # Apply logarithmic scaling
        ])

    def __len__(self):
        return len(self.data_point_names)

    def get_phase(self, idx):
        return np.load(os.path.join(self.master_path, self.data_point_names[idx] + '_phase.npy'))

    # TODO: Try log scaling
    def min_max(self, x):
        """
        Simple min-max normalisation.
        :param x: The unnormalised input.
        :return: The normalised output in [0, 1].
        """
        return (x - np.min(x)) / (np.max(x) - np.min(x))


    def __getitem__(self, idx):
        filename = self.data_point_names[idx]
        label = int(filename[-1])
        phase = self.get_phase(idx)

        #print(np.array(Image.open(os.path.join(self.master_path, self.data_point_names[idx] + '.png'))).mean(axis=-1).shape)

        spectrogram_np = self.min_max(np.array(Image.open(os.path.join(self.master_path, self.data_point_names[idx] + '.png'))).mean(axis=-1))
        spectrogram = self.transforms(spectrogram_np)

        #print(spectrogram.min())
        #print(spectrogram.max())

        return {'spectrogram': spectrogram,
                'label': torch.tensor(label),
                'phase': phase}


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels=None, mask=None):
        device = torch.device('cuda' if features.is_cuda else 'cpu')
        batch_size = features.shape[0]
        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        if labels is not None and mask is None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Number of labels does not match number of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        elif mask is None:
            raise ValueError('If labels are not provided, mask must be provided')

        anchor_feature = contrast_feature
        anchor_count = contrast_count

        # Compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature
        )

        # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Mask out self-contrast cases
        logits_mask = torch.ones_like(mask)
        logits_mask = logits_mask.repeat(contrast_count, contrast_count)

        mask = mask.repeat(contrast_count, contrast_count)

        logits_mask = logits_mask * (1 - torch.eye(logits_mask.shape[0]).to(device))

        # Compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        # Compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # Loss
        loss = - (self.temperature / anchor_count) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


# Define the VAE
class VAE(nn.Module):
    def __init__(self, latent_dim=1024, decoder_channels=[16, 32, 64, 128, 192, 256, 512], use_weight_norm=False):
        super(VAE, self).__init__()
        self.encoder = models.resnet18(weights=None)
        self.encoder.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_output_features = self.encoder.fc.in_features
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-1])  # Remove the final fc layer
        self.fc_mu = nn.Linear(num_output_features, latent_dim)
        self.fc_logvar = nn.Linear(num_output_features, latent_dim)
        self.decoder_channels = decoder_channels

        self.decoder_fc = nn.Linear(latent_dim, decoder_channels[-1] * 16 * 6)

        self.decoder = nn.Sequential()
        for c_i in reversed(range(1, len(decoder_channels))):
            self.decoder.append(DecoderBlock(decoder_channels[c_i], decoder_channels[c_i-1], c_i,
                                             'none', len(decoder_channels),
                                             1024, 384))

        if use_weight_norm:
            self.output = nn.Sequential(
                weight_norm(nn.Conv2d(in_channels=decoder_channels[0],
                                      out_channels=1,
                                      kernel_size=1, stride=1,
                                      padding=0))
            )
        else:
            self.output = nn.Sequential(
                nn.Conv2d(in_channels=decoder_channels[0],
                          out_channels=1,
                          kernel_size=1, stride=1,
                          padding=0)
            )

        # Decoder: Transposed Convolutions
        #self.decoder_fc = nn.Linear(latent_dim, 512 * 7 * 7)  # Adjust dimensions as needed
        #self.decoder_conv = nn.Sequential(
        #    nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1), # Output: (256, 14, 14)
        #    nn.ReLU(),
        #    nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1), # Output: (128, 28, 28)
        #    nn.ReLU(),
        #    nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1), # Output: (64, 56, 56)
        #    nn.ReLU(),
        #    nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), # Output: (32, 112, 112)
        #    nn.ReLU(),
        #    nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1), # Output: (16, 224, 224)
        #    nn.ReLU(),
        #    nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1), # Output: (1, 448, 448)
        #    nn.Sigmoid()  # Output between 0 and 1
        #)

    def encode(self, x):
        encoded = self.encoder(x).squeeze(dim=2).squeeze(dim=2)
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)

        return mu, logvar

    def reparameterise(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_fc(z)
        h = h.view(h.size(0), self.decoder_channels[-1], 16, 6)  # Reshape for the convolutional layers
        h = self.decoder(h)
        h = self.output(h)
        return h  # transforms.Resize((1025, 431))(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar)

        return self.decode(z), mu, logvar


# Training loop for VAE with contrastive learning
def train_vae(data_loader_train, data_loader_val, lr=1e-3, epochs=50, name=None, contrastive_weight=0.01, contrastive_loss=True, visualise=True):

    vae = VAE().to(device)
    optimiser = torch.optim.Adam(vae.parameters(), lr=lr)

    sup_con_loss = SupConLoss()
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_model = None

    for epoch in range(epochs):
        vae.train()
        train_loss = 0
        sdr = 0
        for batch in data_loader_train:
            optimiser.zero_grad()

            spectrograms = batch['spectrogram'].to(device)
            labels = batch['label'].to(device)
            phases = batch['phase'].numpy()

            #print(spectrograms.shape)

            recon, mu, logvar = vae(spectrograms.float())
            recon_loss = criterion(recon.float(), spectrograms.float())
            kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

            z = vae.reparameterise(mu, logvar)
            features = z.unsqueeze(1).repeat(1, 2, 1).to(device)  # SupConLoss expects features with an extra dimension

            supcon_loss_value = sup_con_loss(features, labels=labels) * contrastive_weight if contrastive_loss else 0

            #print(recon_loss)
            #print(kld_loss)
            #print(supcon_loss_value)

            loss = recon_loss + kld_loss + supcon_loss_value
            loss.backward()
            optimiser.step()

            train_loss += loss.item()

            #sdr += compute_spectral_metrics(spectrograms.float(), recon.float(), phases=phases)[metric_index_mapping['sdr']]

        avg_train_loss = train_loss / len(data_loader_train)

        vae.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in data_loader_val:
                spectrograms = batch['spectrogram'].to(device)
                labels = batch['label'].to(device)

                recon, mu, logvar = vae(spectrograms.float())
                recon_loss = criterion(recon.float(), spectrograms.float())
                kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

                z = vae.reparameterise(mu, logvar)
                features = z.unsqueeze(1).repeat(1, 2, 1).to(device)
                supcon_loss_value = sup_con_loss(features, labels=labels) * contrastive_weight if contrastive_loss else 0

                loss = recon_loss + kld_loss + supcon_loss_value
                val_loss += loss.item()

        datapoint = data_loader_val.dataset[4]
        output, _, _ = vae(datapoint['spectrogram'].unsqueeze(dim=0).to(device).float())

        save_spectrogram_to_file(output.squeeze().detach().cpu().numpy(), f'aaa_recon{epoch}.png')

        avg_val_loss = val_loss / len(data_loader_val)
        print(
            f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = vae.state_dict()

            if name:
                torch.save(best_model, f'checkpoints/{name}.pth')
                print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Best model saved as {name} with val loss: {best_val_loss:.4f}")
