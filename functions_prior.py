import copy
import datetime
import json
import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torchaudio
from PIL import Image
from torch.distributions import Normal, kl_divergence
from torch.nn import BCEWithLogitsLoss, L1Loss, MSELoss
from torch.optim.lr_scheduler import CyclicLR
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision import models
from torchvision.models import ResNet18_Weights

from models.cnn_ae_2d_spectrograms import *

from evaluation_metric_functions import compute_spectral_metrics
from functions import metric_index_mapping, save_spectrogram_to_file, get_total_loss
from new_vae import NewVAE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ResNetClassifier(nn.Module):
    def __init__(self, num_classes=4, pretrained=True):
        super(ResNetClassifier, self).__init__()
        self.model = models.resnet18(pretrained=pretrained)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.model(x)
        x = self.softmax(x)
        return x


def compute_sdr(estimated_signal, original_signal, eps=1e-8):
    """
    Calculate the Signal-to-Distortion Ratio (SDR) loss.

    Args:
        estimated_signal (torch.Tensor): The estimated (reconstructed) signal.
        original_signal (torch.Tensor): The original signal.
        eps (float): A small value to avoid division by zero.

    Returns:
        torch.Tensor: The SDR value for each sample.
    """
    # estimated_signal = estimated_signal.view(estimated_signal.size(0), -1)
    # original_signal = original_signal.view(original_signal.size(0), -1)

    if torch.equal(estimated_signal, original_signal):
        return -float('inf')

    num = original_signal ** 2
    denom = (original_signal - estimated_signal) ** 2

    sdr = 10 * (torch.log10(num + eps) - torch.log10(denom + eps))
    return torch.mean(sdr)


class SDRLoss(torch.nn.Module):
    def __init__(self):
        super(SDRLoss, self).__init__()

    def forward(self, estimated_signal, original_signal):
        sdr = compute_sdr(estimated_signal, original_signal)
        return torch.exp(-torch.mean(sdr))  # We negate SDR because we want to maximize it


class PriorDataset(Dataset):
    def __init__(self, split, debug=False, image_h=1024, image_w=384, name='musdb_18_prior', num_stems=4):
        self.data_point_names = []
        self.master_path = os.path.join('data', name, split)

        for data_folder in os.listdir(self.master_path):
            if random.random() < 0.9 and debug:
                continue

            for stem_idx in range(1, 1+num_stems):
                self.data_point_names.append(os.path.join(data_folder, f'{"stems/" if name == "toy_dataset" else ""}stem{stem_idx}'))

        self.transforms = transforms.Compose([
            transforms.ToTensor(),  # Convert numpy array to tensor
            transforms.Resize((image_h, image_w)),
            # TODO: Try normalisation
            # transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize to mean=0.5, std=0.5
            #transforms.Lambda(lambda x: torch.log(x + 1e-9)),  # Apply logarithmic scaling
        ])

    def __len__(self):
        return len(self.data_point_names)

    def get_phase(self, idx):
        return np.load(os.path.join(self.master_path, self.data_point_names[idx] + '_phase.npy'))

    def min_max(self, x):
        """
        Simple min-max normalisation.
        :param x: The unnormalised input.
        :return: The normalised output in [0, 1].
        """
        #return x
        normalised = (x - np.min(x)) / (np.max(x) - np.min(x))
        return normalised


    def __getitem__(self, idx):
        filename = self.data_point_names[idx]
        label = int(filename[-1])-1
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


def compute_size(size, num_layers, stride, kernel_size, padding):

    for _ in range(num_layers-1):
        size = (size + 2*padding - kernel_size) // stride + 1

    return size

# Define the VAE
class VAE(nn.Module):
    def __init__(self, latent_dim=64, channels=[32, 64, 128, 256, 512], use_blocks=True, image_h=1024, image_w=384, kernel_sizes=None, strides=None):
        super(VAE, self).__init__()
        #self.encoder = models.resnet18(weights=None)
        #self.encoder.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        #num_output_features = self.encoder.fc.in_features
        #self.encoder = nn.Sequential(*list(self.encoder.children())[:-1])  # Remove the final fc layer

        self.image_h = image_h
        self.image_w = image_w

        if kernel_sizes is None:
            kernel_sizes = []

            for _ in range(len(channels)):
                kernel_sizes.append(3)

        if strides is None:
            strides = []

            for _ in range(len(channels)):
                strides.append(1)

        assert len(channels) == len(strides) == len(kernel_sizes)

        if channels[0] != 1:
            channels = [1] + channels

        self.channels = channels

        self.encoder = nn.Sequential()
        for c_i in range(len(channels)-1):
            if use_blocks:
                self.encoder.append(EncoderBlock(channels[c_i], channels[c_i + 1], kernel_size=kernel_sizes[c_i], stride=strides[c_i]))
            else:
                self.encoder.append(nn.Conv2d(channels[c_i], channels[c_i+1], kernel_size=kernel_sizes[c_i], stride=strides[c_i], padding=int((kernel_sizes[c_i]-1)/2)))
                self.encoder.append(nn.ReLU())

        #self.compressed_size_h = 1024 // 2**len(channels) if use_blocks else compute_size(1024, len(channels), 2, 3, 3)
        #self.compressed_size_w = 384 // 2**len(channels) if use_blocks else compute_size(384, len(channels), 2, 3, 3)

        dummy_encoded = self.encoder(torch.zeros((1, 1, image_h, image_w)))
        self.compressed_size_h = dummy_encoded.shape[2]
        self.compressed_size_w = dummy_encoded.shape[3]

        num_output_features = channels[-1] * self.compressed_size_h * self.compressed_size_w

        self.fc_mu = nn.Linear(num_output_features, latent_dim)
        self.fc_logvar = nn.Linear(num_output_features, latent_dim)

        self.decoder_fc = nn.Linear(latent_dim, num_output_features)

        self.decoder = nn.Sequential()
        for c_i in reversed(range(1, len(channels))):
            if use_blocks:
                self.decoder.append(DecoderBlock(channels[c_i], channels[c_i - 1], c_i,
                                             'none', len(channels),
                                             image_h, image_w, kernel_size=kernel_sizes[c_i-1], stride=strides[c_i-1]))
            else:
                self.decoder.append(nn.ConvTranspose2d(channels[c_i], channels[c_i-1], kernel_size=kernel_sizes[c_i-1], stride=strides[c_i-1], padding=int((kernel_sizes[c_i-1]-1)/2)))
                self.decoder.append(nn.ReLU())

    def encode(self, x):
        encoded = self.encoder(x)#.squeeze(dim=2).squeeze(dim=2)
        encoded_flat = encoded.flatten(start_dim=1, end_dim=-1)
        mu = self.fc_mu(encoded_flat)
        logvar = self.fc_logvar(encoded_flat)

        return mu, logvar

    def reparameterise(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_fc(z)
        h = h.view(h.size(0), self.channels[-1], self.compressed_size_h, self.compressed_size_w)  # Reshape for the convolutional layers
        h = self.decoder(h)
        #h = self.output(h)
        return transforms.Resize((self.image_h, self.image_w))(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar)

        return self.decode(z), mu, logvar

    def log_prob(self, x):
        recon, mu, logvar = self.forward(x)

        # KL divergence
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss_fn = MSELoss(reduction='sum')
        recon_loss = loss_fn(recon, x)

        # ELBO as approximation of log-probability
        elbo = -(recon_loss + kl_div)
        return elbo / x.size(0)



class LatentClassifier(nn.Module):
    def __init__(self, latent_dim=4096):
        super(LatentClassifier, self).__init__()
        self.fc1 = nn.Linear(latent_dim, latent_dim//2)
        self.fc2 = nn.Linear(latent_dim//2, latent_dim//4)
        self.fc3 = nn.Linear(latent_dim//4, 4)

    def forward(self, z):
        x = torch.relu(self.fc1(z))
        x = torch.relu(self.fc2(x))
        ddd = self.fc3(x)
        x = torch.softmax(self.fc3(x), dim=1)

        return x


def train_classifier(data_loader_train, data_loader_val, vae, lr=1e-3, epochs=50, name=None, naive=False, pretrained=True):
    model = ResNetClassifier(pretrained=pretrained).to(device) if naive else LatentClassifier().to(device)

    if not naive:
        vae = vae.to(device)
        vae.eval()

    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        total_train = 0
        correct_train = 0

        total_val = 0
        correct_val = 0

        best_model = None

        for batch in data_loader_train:
            optimiser.zero_grad()

            spectrograms = batch['spectrogram'].to(device)
            labels = batch['label'].to(device)
            phases = batch['phase'].numpy()

            if naive:
                output = model(spectrograms.float())
            else:
                with torch.no_grad():
                    mu, logvar = vae.encode(spectrograms.float())
                    z = vae.reparameterise(mu, logvar)

                output = model(z)

            loss = criterion(output, labels)
            loss.backward()
            optimiser.step()

            train_loss += loss.item()

            _, predicted = torch.max(output.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        avg_train_loss = train_loss / len(data_loader_train.dataset)

        model.eval()
        val_loss = 0

        with torch.no_grad():
            for batch in data_loader_val:
                spectrograms = batch['spectrogram'].to(device)
                labels = batch['label'].to(device)

                if naive:
                    output = model(spectrograms.float())
                else:
                    mu, logvar = vae.encode(spectrograms.float())
                    z = vae.reparameterise(mu, logvar)

                    output = model(z)

                loss = criterion(output, labels)

                val_loss += loss.item()

                _, predicted = torch.max(output.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(data_loader_val.dataset)

        if avg_val_loss < best_val_loss:
            best_model = model
            best_val_loss = avg_val_loss

            if name:
                torch.save(best_model.state_dict(), f'checkpoints/{name}.pth')
                print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Best model saved as {name} with val loss: {best_val_loss:.4f}")


        print(
            f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        print(f'Train accuracy: {100 * correct_train / total_train:.2f}%, Val accuracy: {100 * correct_val / total_val:.2f}%')

def export_hyperparameters_to_file(name, channels, hidden, kernel_sizes, strides, use_blocks, contrastive_weight, contrastive_loss, kld_weight, image_h, image_w, sigma):
    """
    Saves the passed hyperparameters to a json file.
    :return: None
    """
    variables = {
        'name': name,
        'channels': channels,
        'hidden': hidden,
        'kernel_sizes': kernel_sizes,
        'strides': strides,
        'use_blocks': use_blocks,
        'contrastive_weight': contrastive_weight,
        'contrastive_loss': contrastive_loss,
        'kld_weight': kld_weight,
        'image_h': image_h,
        'image_w': image_w,
        'sigma': sigma
    }

    if not os.path.exists('hyperparameters'):
        os.mkdir('hyperparameters')

    with open(f'hyperparameters/{name}.json', 'w') as file:
        json.dump(variables, file)


def train_vae(data_loader_train, data_loader_val, lr=1e-3, use_blocks=False, epochs=50, latent_dim=64, criterion=nn.L1Loss(), name=None, contrastive_weight=0.01, contrastive_loss=True, visualise=True, channels=[32, 64, 128, 256, 512], kld_weight=0.0001, recon_weight=1, verbose=True, image_h=1024, image_w=384, cyclic_lr=False, kernel_sizes=None, strides=None, sigma=False):
    print(f'Training {name}')

    criterion.reduction = 'sum'

    export_hyperparameters_to_file(name, channels, latent_dim, kernel_sizes, strides, use_blocks, contrastive_weight, contrastive_loss, kld_weight, image_h, image_w, sigma)

    vae = VAE(use_blocks=use_blocks, latent_dim=latent_dim, channels=channels, kernel_sizes=kernel_sizes, strides=strides, image_h=image_h, image_w=image_w).to(device)

    optimiser = torch.optim.Adam(vae.parameters(), lr=lr)

    if cyclic_lr:
        scheduler = CyclicLR(optimiser, base_lr=lr*0.1, max_lr=lr*10, step_size_up=2000, mode='triangular')

    sup_con_loss = SupConLoss()
    #criterion = nn.MSELoss()
    #criterion = SDRLoss()
    #criterion = BCEWithLogitsLoss()

    best_val_loss = float('inf')
    best_model = None
    best_val_sdr = -float('inf')

    for epoch in range(epochs):
        vae.train()
        train_loss = 0
        for idx, batch in enumerate(data_loader_train):
            optimiser.zero_grad()

            spectrograms = batch['spectrogram'].to(device)
            labels = batch['label'].to(device)

            if sigma is not None:
                epsilon = torch.randn(spectrograms.shape).to(device) * sigma**2
                spectrograms = spectrograms + epsilon

            #print(spectrograms.shape)

            recon, mu, logvar = vae(spectrograms.float())
            recon_loss = criterion(recon.float(), spectrograms.float()) * recon_weight
            kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) * kld_weight

            z = vae.reparameterise(mu, logvar)
            features = z.unsqueeze(1).repeat(1, 2, 1).to(device)  # SupConLoss expects features with an extra dimension

            supcon_loss_value = sup_con_loss(features, labels=labels) * contrastive_weight if contrastive_loss else 0

            #print(recon_loss)
            #print(kld_loss)
            #print(supcon_loss_value)

            loss = recon_loss + kld_loss + supcon_loss_value

            loss.backward()
            optimiser.step()

            if cyclic_lr:
                scheduler.step()

            train_loss += loss.item()

            #sdr += compute_spectral_metrics(spectrograms.float(), recon.float(), phases=phases)[metric_index_mapping['sdr']]

        avg_train_loss = train_loss / len(data_loader_train.dataset)

        epoch_val_sdr = 0

        vae.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in data_loader_val:
                spectrograms = batch['spectrogram'].to(device)
                labels = batch['label'].to(device)

                if sigma is not None:
                    spectrograms = spectrograms + torch.randn(spectrograms.shape).to(device) * sigma ** 2

                recon, mu, logvar = vae(spectrograms.float())
                recon_loss = criterion(recon.float(), spectrograms.float()) * recon_weight
                kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) * kld_weight

                z = vae.reparameterise(mu, logvar)
                features = z.unsqueeze(1).repeat(1, 2, 1).to(device)
                supcon_loss_value = sup_con_loss(features, labels=labels) * contrastive_weight if contrastive_loss else 0

                loss = recon_loss + kld_loss + supcon_loss_value
                val_loss += loss.item()
                epoch_val_sdr += -torch.log(recon_loss)

        avg_val_loss = val_loss / len(data_loader_val.dataset)

        val_sdr = epoch_val_sdr / len(data_loader_val.dataset)

        if val_sdr > best_val_sdr:
            best_val_sdr = val_sdr

        if visualise:
            datapoint = data_loader_val.dataset[9]
            output, _, _ = vae(datapoint['spectrogram'].unsqueeze(dim=0).to(device).float())

            save_spectrogram_to_file(output.squeeze().detach().cpu().numpy(), f'{name}_{epoch}.png')

            if epoch == 0:
                spectrogram = datapoint['spectrogram'].squeeze()
                if sigma is not None:
                    spectrogram = spectrogram + torch.randn(spectrogram.shape) * sigma**2
                save_spectrogram_to_file(spectrogram, f'{name}_gt.png')

        if verbose:
            print(
                f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = vae

            if name:
                torch.save(best_model.state_dict(), f'checkpoints/{name}.pth')
                print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Best model saved as {name} with val loss: {best_val_loss:.4f}")

    return best_model, best_val_sdr


def finetune_sigma(og_vae, dataloader_train, dataloader_val, sigma, criterion=nn.MSELoss(), lr=1e-05, epochs=10, verbose=False, visualise=False, recon_weight=1, kld_weight=1):

    criterion.reduction = 'sum'

    vae = og_vae # copy.deepcopy(og_vae)
    optimiser = torch.optim.Adam(vae.parameters(), lr=lr)

    best_val_loss = float('inf')
    best_model = None

    for epoch in range(epochs):
        vae.train()
        train_loss = 0
        for idx, batch in enumerate(dataloader_train):
            optimiser.zero_grad()

            spectrograms_clean = batch['spectrogram'].float()
            epsilon = torch.randn(spectrograms_clean.shape) * sigma**2
            spectrograms_sigma = spectrograms_clean + epsilon
            spectrograms_sigma = spectrograms_sigma.float().to(device)

            recon, mu, logvar = vae(spectrograms_sigma)
            recon_loss = criterion(recon, spectrograms_sigma) * recon_weight
            kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) * kld_weight

            loss = recon_loss + kld_loss

            loss.backward()
            optimiser.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(dataloader_train.dataset)

        vae.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in dataloader_val:
                spectrograms_clean = batch['spectrogram'].float()
                spectrograms_sigma = spectrograms_clean + torch.randn(spectrograms_clean.shape) * sigma ** 2
                spectrograms_sigma = spectrograms_sigma.float().to(device)

                recon, mu, logvar = vae(spectrograms_sigma)
                recon_loss = criterion(recon, spectrograms_sigma) * recon_weight
                kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) * kld_weight

                loss = recon_loss + kld_loss
                val_loss += loss.item()

        avg_val_loss = val_loss / len(dataloader_val.dataset)


        if visualise:
            datapoint = dataloader_val.dataset[9]
            spectrogram = datapoint['spectrogram'].unsqueeze(dim=0)
            epsilon = torch.randn(spectrogram.shape) * sigma**2
            spectrogram = spectrogram + epsilon
            output, _, _ = vae(spectrogram.to(device).float())

            save_spectrogram_to_file(output.squeeze().detach().cpu().numpy(), f'sigma_{np.round(sigma, 3)}_{epoch}.png')

            if epoch == 0:
                save_spectrogram_to_file(spectrogram.squeeze(), f'sigma_{np.round(sigma, 3)}_gt.png')

        if verbose:
            print(
                f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = vae

            if visualise:
                torch.save(best_model.state_dict(), f'checkpoints/sigma_{np.round(sigma, 3)}.pth')
                print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Best model saved as sigma_{np.round(sigma, 3)} with val loss: {best_val_loss:.4f}")



    return best_model
