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

from evaluation_metric_functions import compute_spectral_metrics
from functions import metric_index_mapping

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
            transforms.Resize((1024, 384)),
            # TODO: Try normalisation
            #transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize to mean=0.5, std=0.5
            #transforms.Lambda(lambda x: torch.log(x + 1e-9)),  # Apply logarithmic scaling
        ])

    def __len__(self):
        return len(self.data_point_names)

    def get_phase(self, idx):
        return np.load(os.path.join(self.master_path, self.data_point_names[idx] + '_phase.npy'))

    def __getitem__(self, idx):
        filename = self.data_point_names[idx]
        label = int(filename[-1])
        phase = self.get_phase(idx)

        #print(np.array(Image.open(os.path.join(self.master_path, self.data_point_names[idx] + '.png'))).mean(axis=-1).shape)

        spectrogram = self.transforms(np.array(Image.open(os.path.join(self.master_path, self.data_point_names[idx] + '.png'))).mean(axis=-1)) / 255

        #print(spectrogram.min())
        #print(spectrogram.max())

        return {'spectrogram': spectrogram,
                'label': torch.tensor(label),
                'phase': phase}



class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf."""
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels):
        """Compute loss for model.

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
        Returns:
            A loss scalar.
        """
        device = features.device

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...], at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')

        mask = torch.eq(labels, labels.T).float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point.
        # Edge case e.g.:-
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan]
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


# Define the VAE
class VAE(nn.Module):
    def __init__(self, latent_dim=128):
        super(VAE, self).__init__()
        self.encoder = models.resnet18(pretrained=False)
        self.encoder.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.encoder.fc = nn.Linear(self.encoder.fc.in_features, latent_dim * 2)

        self.decoder_fc = nn.Linear(latent_dim, 512 * 16 * 6)
        # Decoder: Transposed Convolutions
        #self.decoder_fc = nn.Linear(latent_dim, 512 * 7 * 7)  # Adjust dimensions as needed
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1), # Output: (256, 14, 14)
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1), # Output: (128, 28, 28)
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1), # Output: (64, 56, 56)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), # Output: (32, 112, 112)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1), # Output: (16, 224, 224)
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1), # Output: (1, 448, 448)
            nn.Sigmoid()  # Output between 0 and 1
        )

    def encode(self, x):
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=1)
        return mu, logvar

    def reparameterise(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_fc(z)
        h = h.view(h.size(0), 512, 16, 6)  # Reshape for the convolutional layers
        h = self.decoder_conv(h)
        return h# transforms.Resize((1025, 431))(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar)
        return self.decode(z), mu, logvar


# Training loop for VAE with contrastive learning
def train_vae(data_loader_train, data_loader_val, lr=1e-3, epochs=50, name=None):

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

            contrastive_weight = 0.01
            supcon_loss_value = sup_con_loss(features, labels) * contrastive_weight

            #print(recon_loss)
            #print(kld_loss)
            #print(supcon_loss_value)

            loss = recon_loss + kld_loss# + supcon_loss_value
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
                supcon_loss_value = sup_con_loss(features, labels)

                loss = recon_loss + kld_loss# + supcon_loss_value
                val_loss += loss.item()

        avg_val_loss = val_loss / len(data_loader_val)
        print(
            f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = vae.state_dict()

            if name:
                torch.save(best_model, f'checkpoints/{name}.pth')
                print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Best model saved as {name} with val loss: {best_val_loss:.4f}")
