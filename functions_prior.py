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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PriorDataset(Dataset):
    def __init__(self, split, debug=False):
        self.data_point_names = []
        self.master_path = os.path.join('data', 'musdb_18_prior', split)

        for data_folder in os.listdir(self.master_path):
            if random.random() < 0.99 and debug:
                continue

            for stem_idx in range(1, 5):
                self.data_point_names.append(os.path.join(data_folder, f'stem{stem_idx}.png'))

        self.transforms = transforms.Compose([
            transforms.ToTensor(),  # Convert numpy array to tensor
            # TODO: Try normalisation
            #transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize to mean=0.5, std=0.5
            #transforms.Lambda(lambda x: torch.log(x + 1e-9)),  # Apply logarithmic scaling
        ])

    def __len__(self):
        return len(self.data_point_names)

    def get_phase(self, idx):
        return np.load(os.path.join(self.master_path, self.data_point_names[idx]))

    def __getitem__(self, idx):
        filename = self.data_point_names[idx]
        label = int(filename[-5])

        return {'spectrogram': self.transforms(np.array(Image.open(os.path.join(self.master_path, self.data_point_names[idx])))),
                'label': label}



class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        anchor_feature = contrast_feature
        anchor_count = contrast_count

        logits = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(mask.shape[0]).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = -mean_log_prob_pos
        loss = loss.view(anchor_count, contrast_count).mean()

        return loss


# Define the VAE
class VAE(nn.Module):
    def __init__(self, latent_dim=128):
        super(VAE, self).__init__()
        self.encoder = models.resnet18(pretrained=False)
        self.encoder.fc = nn.Linear(self.encoder.fc.in_features, latent_dim * 2)

        self.decoder_fc = nn.Linear(latent_dim, self.encoder.fc.in_features)
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
        h = h.view(h.size(0), 512, 7, 7)  # Reshape for the convolutional layers
        h = self.decoder_conv(h)
        return h

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar)
        return self.decode(z), mu, logvar


# Training loop for VAE with contrastive learning
def train_vae(data_loader_train, data_loader_val, lr=1e-3, epochs=50):

    vae = VAE().to(device)
    optimiser = torch.optim.Adam(vae.parameters(), lr=lr)

    sup_con_loss = SupConLoss()

    criterion = nn.MSELoss()

    vae.train()
    for epoch in range(epochs):
        for batch in data_loader_train:
            optimiser.zero_grad()

            spectrograms = batch['spectrogram']
            labels = batch['label']

            recon, mu, logvar = vae(spectrograms)
            recon_loss = criterion(recon, spectrograms)
            kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

            z = vae.reparameterize(mu, logvar)
            features = z.unsqueeze(1).repeat(1, 2, 1)  # SupConLoss expects features with an extra dimension
            supcon_loss_value = sup_con_loss(features, labels)

            loss = recon_loss + kld_loss + supcon_loss_value
            loss.backward()
            optimiser.step()

