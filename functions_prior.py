import datetime
import glob
import json
import math
import os
import pickle
import random

import cv2
import mir_eval.separation
import numpy as np
import torch
import torch.nn as nn
import torchaudio
from PIL import Image
from torch.distributions import Normal, kl_divergence
from torch.nn import BCEWithLogitsLoss, L1Loss, MSELoss, BCELoss
from torch.optim.lr_scheduler import CyclicLR
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision import models
from torchvision.io import read_video
from torchvision.models import ResNet18_Weights
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large, raft_small, Raft_Small_Weights
from torchvision.models.video import R3D_18_Weights
from torchvision.utils import save_image

from models.cnn_ae_2d_spectrograms import *

from evaluation_metric_functions import compute_spectral_metrics
from functions import metric_index_mapping, save_spectrogram_to_file, get_total_loss
from new_vae import NewVAE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MultiModalDataset(Dataset):
    def __init__(self, split, image_h=64, image_w=64, video_h=128, video_w=128, debug=False, normalise=False, fps=30):
        self.video_master_path = os.path.join('data', 'rochester_preprocessed', split)
        self.datapoints = os.listdir(self.video_master_path)[::2] if not debug else os.listdir(self.video_master_path)[::40]

        self.video_h = video_h
        self.video_w = video_w
        self.image_h = image_h
        self.image_w = image_w

        self.video_transforms = transforms.Compose([
            # transforms.ToPILImage(),  # Ensure this if the input is a NumPy array
            transforms.Resize((video_h, video_w)),
            # transforms.ToTensor(),
            # transforms.Lambda(lambda x: (x / 255.0) * 2.0 - 1.0),
        ])

        self.image_transforms = transforms.Compose([
            # transforms.ToPILImage(),  # Ensure this if the input is a NumPy array
            transforms.Resize((image_h, image_w)),
            transforms.ToTensor(),
            # transforms.Lambda(lambda x: (x / 255.0) * 2.0 - 1.0),
        ])

        self.debug = debug
        self.normalise = normalise
        self.fps = fps

    def __len__(self):
        return len(self.datapoints) * 2

    def get_phase(self, idx):
        return np.load(os.path.join(self.video_master_path, self.datapoints[idx] + '_phase.npy'))

    def min_max(self, x):
        """
        Simple min-max normalisation.
        :param x: The unnormalised input.
        :return: The normalised output in [0, 1].
        """
        #return x
        normalised = (x - torch.min(x)) / (torch.max(x) - torch.min(x))
        return normalised


    def __getitem__(self, idx):
        filename = os.path.join(self.video_master_path, self.datapoints[idx // 2])

        video_path = os.path.join(filename, 'video.mp4')

        video_tensor = read_video(video_path, output_format='TCHW', pts_unit='sec')[0]
        # video_tensor = self.video_transforms(video_tensor)

        # print(video_tensor.shape)

        transformed_frames = []

        skips = math.floor(30 / self.fps)
        new_num_frames = 150 // skips

        for frame_idx in range(video_tensor.shape[0]):
            if frame_idx % skips == 0:
                frame = video_tensor[frame_idx]
                if self.fps < 30:
                    frame = frame[:, 500:900, :]
                frame = self.video_transforms(frame).float()  # Apply the defined transformations

                if self.normalise:
                    transformed_frames.append(frame / 255)
                else:
                    transformed_frames.append(frame)

        video_tensor = torch.stack(transformed_frames, dim=0)

        # (num_frames, 3, h, w)

        num_frames = video_tensor.shape[0]
        if num_frames < new_num_frames:
            # Calculate how many frames to add
            frames_to_add = new_num_frames - num_frames

            # Create black frames with the same shape as existing frames (C, H, W)
            black_frame = torch.zeros((self.video_h, self.video_w, 3)).permute(2, 0, 1)  # (C, H, W)
            black_frames = black_frame.unsqueeze(0).repeat(frames_to_add, 1, 1, 1)  # (frames_to_add, C, H, W)

            # Concatenate the original video tensor with the black frames
            video_tensor = torch.cat([video_tensor, black_frames], dim=0)

        # (num_frames, 3, h, w)
        assert video_tensor.shape[0] == new_num_frames, "Video tensor should have exactly 150 frames after padding."

        if idx % 2 == 0:
            png_files = glob.glob(f'{filename}/*.png')
            # this means the video matches the audio
            label = 1
            source_files = [png_files[i] for i in range(2)]
            # order shouldn't matter
        else:
            # this means the video and audio don't match
            label = 0
            filename1 = os.path.join(self.video_master_path, self.datapoints[random.randint(0, len(self.datapoints)-1)])
            filename2 = os.path.join(self.video_master_path, self.datapoints[random.randint(0, len(self.datapoints)-1)])
            filenames = [filename1, filename2]

            png_files1 = glob.glob(f'{filename1}/*.png')
            png_files2 = glob.glob(f'{filename2}/*.png')
            png_files = [png_files1, png_files2]

            source_files = [png_files[i][i] for i in range(2)]

        random.shuffle(source_files)

        spectrograms = [Image.open(source_files[i]).convert('L') for i in range(2)]
        # spectrograms = [self.min_max(spectrograms[i]) for i in range(2)]
        spectrograms = [self.image_transforms(spectrograms[i]).squeeze() for i in range(2)]

        if self.normalise:
            spectrograms = [self.min_max(spectrogram) for spectrogram in spectrograms]

        spectrograms = torch.stack(spectrograms, dim=0)

        #print(spectrogram.min())
        #print(spectrogram.max())

        _indices = [file.rindex('_')+1 for file in source_files]
        dot_indices = [index+2 for index in _indices]

        stem_names = [source_files[i][_indices[i]:dot_indices[i]] for i in range(2)]

        return {
                'video': video_tensor,
                'sources': spectrograms,
                'label': torch.tensor(label, dtype=torch.float32),
                'stem_names': stem_names
               }

def save_tensor_image_to_png(image_tensor, filename):
    image_tensor = image_tensor.permute(1, 2, 0)

    # Convert the tensor to a NumPy array and scale to [0, 255]
    # image_array = (image_tensor.numpy() * 255).astype('uint8')

    # Create a PIL image

    image_array = (image_tensor.numpy()).astype('uint8')

    image = Image.fromarray(image_array)

    # Save the image as PNG
    image.save(filename)


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


class PriorDatasetVideo(Dataset):
    def __init__(self, split, debug=False, image_h=64, image_w=64, sigma=None, stem_type=None):
        self.data_point_names = []
        self.master_path = os.path.join('data', 'rochester_preprocessed', split)
        self.sigma = sigma
        self.stem_type = stem_type

        # permitted_types = list(range(1, num_stems+1)) if stem_type is None else [stem_type]

        for data_folder in os.listdir(self.master_path):
            if random.random() < 0.9 and debug:
                continue

            folder = os.path.join(self.master_path, data_folder)

            for file in os.listdir(folder):
                if file.__contains__(stem_type) and file[-4:] == '.png':
                    self.data_point_names.append(os.path.join(data_folder, file[:-4]))
                    break

        self.transforms = transforms.Compose([
            transforms.ToTensor(),  # Convert numpy array to tensor
            transforms.Resize((image_h, image_w)),
            # transforms.Lambda(lambda x: (x / 255.0) * 2.0 - 1.0)
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

        #print(np.array(Image.open(os.path.join(self.master_path, self.data_point_names[idx] + '.png'))).mean(axis=-1).shape)

        if idx > len(self.data_point_names)-1:
            print('Index too high.')
            idx = len(self.data_point_names)-1

        unnormalised_spectrogram = np.array(Image.open(os.path.join(self.master_path, self.data_point_names[idx] + '.png'))).mean(axis=-1)

        if self.sigma is not None:
            unnormalised_spectrogram += np.random.randn(*unnormalised_spectrogram.shape) * self.sigma**2

        spectrogram_np = self.min_max(unnormalised_spectrogram)
        spectrogram = self.transforms(spectrogram_np)

        # print(spectrogram.min())
        # print(spectrogram.max())

        return {'spectrogram': spectrogram}


class PriorDataset(Dataset):
    def __init__(self, split, debug=False, image_h=1024, image_w=384, name='musdb_18_prior', num_stems=4, sigma=None, stem_type=None):
        self.data_point_names = []
        self.master_path = os.path.join('data', name, split)
        self.sigma = sigma
        self.stem_type = stem_type

        permitted_types = list(range(1, num_stems+1)) if stem_type is None else [stem_type]

        for data_folder in os.listdir(self.master_path):
            if random.random() < 0.9 and debug:
                continue

            for stem_idx in range(1, 1+num_stems):
                if permitted_types.__contains__(stem_idx):
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

        unnormalised_spectrogram = np.array(Image.open(os.path.join(self.master_path, self.data_point_names[idx] + '.png'))).mean(axis=-1)

        if self.sigma is not None:
            unnormalised_spectrogram += np.random.randn(*unnormalised_spectrogram.shape) * self.sigma**2

        spectrogram_np = self.min_max(unnormalised_spectrogram)
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
    def __init__(self, latent_dim=64, channels=[32, 64, 128, 256, 512], use_blocks=True, image_h=1024, image_w=384, kernel_sizes=None, strides=None, batch_norm=False, num_input_channels=1):
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
            channels = [num_input_channels] + channels

        self.channels = channels

        self.encoder = nn.Sequential()
        for c_i in range(len(channels)-1):
            if use_blocks:
                self.encoder.append(EncoderBlock(channels[c_i], channels[c_i + 1], kernel_size=kernel_sizes[c_i], stride=strides[c_i]))
            else:
                self.encoder.append(nn.Conv2d(channels[c_i], channels[c_i+1], kernel_size=kernel_sizes[c_i], stride=strides[c_i], padding=int((kernel_sizes[c_i]-1)/2)))

                if batch_norm:
                    self.encoder.append(nn.BatchNorm2d(channels[c_i+1]))

                self.encoder.append(nn.ReLU())

        #self.compressed_size_h = 1024 // 2**len(channels) if use_blocks else compute_size(1024, len(channels), 2, 3, 3)
        #self.compressed_size_w = 384 // 2**len(channels) if use_blocks else compute_size(384, len(channels), 2, 3, 3)

        dummy_encoded = self.encoder(torch.zeros((1, num_input_channels, image_h, image_w)))
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

                if c_i > 1:
                    if batch_norm:
                        self.decoder.append(nn.BatchNorm2d(channels[c_i-1]))
                    self.decoder.append(nn.ReLU())
                else:
                    # Add after last layer without batch norm
                    self.decoder.append(nn.Sigmoid())

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

    def latent_forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar)

        return z, mu, logvar

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar)

        return self.decode(z), mu, logvar

    def log_prob(self, x, normalise=True):
        recon, mu, logvar = self.forward(x)

        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss_fn = MSELoss(reduction='sum')
        recon_loss = loss_fn(recon, x)

        elbo = -(recon_loss + kl_div)
        return elbo / x.size(0) if normalise else elbo



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


def export_hyperparameters_to_file_video(name, z_dim_2d, z_dim_3d, video_h, video_w, image_h, image_w, normalise, fps):
    """
    Saves the passed hyperparameters to a json file.
    :return: None
    """
    variables = {
        'name': name,
        'z_dim_2d': z_dim_2d,
        'z_dim_3d': z_dim_3d,
        'video_h': video_h,
        'video_w': video_w,
        'image_h': image_h,
        'image_w': image_w,
        'normalise': normalise,
        'fps': fps
    }

    if not os.path.exists('hyperparameters'):
        os.mkdir('hyperparameters')

    with open(f'hyperparameters/{name}.json', 'w') as file:
        json.dump(variables, file)



def train_vae(data_loader_train, data_loader_val, lr=1e-3, use_blocks=False, epochs=50, latent_dim=64, criterion=nn.MSELoss(), name=None, contrastive_weight=0.01, contrastive_loss=False, visualise=True, channels=[32, 64, 128, 256, 512], kld_weight=1, recon_weight=1, verbose=True, image_h=1024, image_w=384, cyclic_lr=False, kernel_sizes=None, strides=None, batch_norm=False, sigma=None, finetune=False):
    # print(f'Training {name}')

    criterion.reduction = 'sum'

    if sigma is not None:
        data_loader_train.dataset.sigma = sigma
        data_loader_val.dataset.sigma = sigma

    export_hyperparameters_to_file(name, channels, latent_dim, kernel_sizes, strides, use_blocks, contrastive_weight, contrastive_loss, kld_weight, image_h, image_w, sigma)

    vae = VAE(use_blocks=use_blocks, latent_dim=latent_dim, channels=channels, kernel_sizes=kernel_sizes,
                  strides=strides, image_h=image_h, image_w=image_w, batch_norm=batch_norm).to(device)

    if finetune:
        vae.load_state_dict(torch.load(f'checkpoints/{name}.pth'))

    optimiser = torch.optim.Adam(vae.parameters(), lr=lr)

    if cyclic_lr:
        scheduler = CyclicLR(optimiser, base_lr=lr*0.1, max_lr=lr*10, step_size_up=2000, mode='triangular')

    sup_con_loss = SupConLoss()
    #criterion = nn.MSELoss()
    #criterion = SDRLoss()
    #criterion = BCEWithLogitsLoss()

    best_val_loss = float('inf')
    best_recon_loss = float('inf')
    best_model = None
    best_val_sdr = -float('inf')

    for epoch in range(epochs):
        vae.train()
        train_loss = 0
        sdr = 0
        for idx, batch in enumerate(data_loader_train):
            optimiser.zero_grad()

            if type(batch) is dict:
                spectrograms = batch['spectrogram'].to(device)
            else:
                spectrograms = batch[0].to(device)
                labels = batch[1].to(device)
            #print(spectrograms.shape)

            recon, mu, logvar = vae(spectrograms.float())
            recon_loss = criterion(recon.float(), spectrograms.float()) * recon_weight
            kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) * kld_weight

            z = vae.reparameterise(mu, logvar)
            features = z.unsqueeze(1).repeat(1, 2, 1).to(device)  # SupConLoss expects features with an extra dimension

            #  = sup_con_loss(features, labels=labels) * contrastive_weight if contrastive_loss else 0

            #print(recon_loss)
            #print(kld_loss)
            #print(supcon_loss_value)

            loss = recon_loss + kld_loss

            loss.backward()
            optimiser.step()

            if cyclic_lr:
                scheduler.step()

            train_loss += loss.item()

            #sdr += compute_spectral_metrics(spectrograms.float(), recon.float(), phases=phases)[metric_index_mapping['sdr']]

        avg_train_loss = train_loss / len(data_loader_train.dataset)

        epoch_val_sdr = 0
        epoch_recon_loss = 0

        vae.eval()
        val_loss = 0
        epoch_recon_loss = 0
        with torch.no_grad():
            for batch in data_loader_val:

                if type(batch) is dict:
                    spectrograms = batch['spectrogram'].to(device)
                else:
                    spectrograms = batch[0].to(device)

                recon, mu, logvar = vae(spectrograms.float())
                recon_loss = criterion(recon.float(), spectrograms.float()) * recon_weight
                kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) * kld_weight

                z = vae.reparameterise(mu, logvar)
                features = z.unsqueeze(1).repeat(1, 2, 1).to(device)

                loss = recon_loss + kld_loss
                val_loss += loss.item()
                epoch_val_sdr += -torch.log(recon_loss)
                epoch_recon_loss += criterion(recon.float(), spectrograms.float()) / len(batch)

        avg_val_loss = val_loss / len(data_loader_val.dataset)
        val_sdr = epoch_val_sdr / len(data_loader_val.dataset)
        epoch_recon_loss = epoch_recon_loss / len(data_loader_val.dataset)

        if val_sdr > best_val_sdr:
            best_val_sdr = val_sdr

        if epoch_recon_loss < best_recon_loss:
            best_recon_loss = epoch_recon_loss

        if visualise:
            datapoint = data_loader_val.dataset[9]

            if type(datapoint) is dict:
                spectrogram = datapoint['spectrogram']
            else:
                spectrogram = datapoint[0]

            output, _, _ = vae(spectrogram.unsqueeze(dim=0).to(device).float())

            save_spectrogram_to_file(output.squeeze().detach().cpu().numpy(), f'{name}_{epoch}.png')

            if epoch == 0:
                spectrogram = spectrogram.squeeze()
                if sigma is not None:
                    spectrogram = spectrogram + torch.randn(spectrogram.shape) * sigma**2
                save_spectrogram_to_file(spectrogram, f'{name}_gt.png')
                save_spectrogram_to_file(spectrogram.squeeze(), f'{name}_gt.png')

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


def finetune_sigma(og_vae, dataloader_train, dataloader_val, sigma, criterion=nn.MSELoss(), lr=1e-05, epochs=10, verbose=False, visualise=False, recon_weight=1, kld_weight=1, parent_name=None):

    stem_type = dataloader_train.dataset.stem_type
    name = f'sigma_{parent_name}_stem{stem_type}_{np.round(sigma, 3)}'
    print(f'Fine-tuning {name}')
    criterion.reduction = 'sum'

    vae = og_vae # copy.deepcopy(og_vae)
    optimiser = torch.optim.Adam(vae.parameters(), lr=lr)

    if sigma is not None:
        dataloader_train.dataset.sigma = sigma
        dataloader_val.dataset.sigma = sigma

    best_val_loss = float('inf')
    best_model = None

    for epoch in range(epochs):
        vae.train()
        train_loss = 0
        for idx, batch in enumerate(dataloader_train):
            optimiser.zero_grad()

            spectrograms = batch['spectrogram'].float().to(device)

            recon, mu, logvar = vae(spectrograms)
            recon_loss = criterion(recon, spectrograms) * recon_weight
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
                spectrograms = batch['spectrogram'].float().to(device)

                recon, mu, logvar = vae(spectrograms)
                recon_loss = criterion(recon, spectrograms) * recon_weight
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

            save_spectrogram_to_file(output.squeeze().detach().cpu().numpy(), f'{name}_{epoch}.png')

            if epoch == 0:
                save_spectrogram_to_file(spectrogram.squeeze(), f'{name}_gt.png')

        if verbose:
            print(
                f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = vae

            if visualise:
                torch.save(best_model.state_dict(), f'checkpoints/{name}.pth')
                print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Best model saved as {name} with val loss: {best_val_loss:.4f}")



    return best_model

def test_vae(vae, dataset, num_samples=64):

    dataloader = DataLoader(dataset, shuffle=True, batch_size=num_samples)

    images_gt = next(iter(dataloader))['spectrogram']
    images_recon, _, _ = vae(images_gt.float().to(device))

    total_sdr = 0

    for i in range(num_samples):

        sdr, isr, sir, sar, _ = mir_eval.separation.bss_eval_images(images_gt[i].view(-1).detach().cpu(),
                                                                    images_recon[i].view(-1).detach().cpu())

        total_sdr += sdr

    return total_sdr / num_samples


class ResNetVAE(nn.Module):
    def __init__(self, original_resnet, latent_dim, deterministic=False):
        super(ResNetVAE, self).__init__()

        # Retain all layers except the final fully connected layer
        self.resnet = nn.Sequential(*list(original_resnet.children())[:-1])


        # Flattening layer
        self.flatten = nn.Flatten()

        # Fully connected layers for mean and log variance
        self.fc_mu = nn.Linear(original_resnet.fc.in_features, latent_dim)
        self.fc_logvar = nn.Linear(original_resnet.fc.in_features, latent_dim)

        self.latent_dim = latent_dim

    def encode(self, x):
        # Pass through ResNet
        x = self.resnet(x)

        # Flatten the output for the FC layers
        x = self.flatten(x)

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        return mu, logvar

    def reparameterize(self, mu, logvar):
        # Reparameterization trick to sample from N(mu, var) from N(0,1)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

class SimpleFCN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1):
        super(SimpleFCN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # First fully connected layer
        self.relu = nn.ReLU()                         # Activation function
        self.fc2 = nn.Linear(hidden_size, output_size) # Second fully connected layer
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)  # Pass input through first layer
        out = self.relu(out)  # Apply ReLU activation
        out = self.fc2(out)  # Pass through second layer
        return self.sigmoid(out)


class VideoModel(nn.Module):
    def __init__(self, z_dim_2d, z_dim_3d, use_optical_flow=True, use_resnet=True, device=torch.device('cuda')):
        super(VideoModel, self).__init__()

        self.model_raft = raft_small(weights=Raft_Small_Weights.DEFAULT)
        self.model_raft.eval()

        resnet_3d = models.video.r3d_18(weights=R3D_18_Weights.DEFAULT)
        resnet_3d.stem[0] = nn.Conv3d(2 if use_optical_flow else 3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)

        self.vae_resnet_3d = ResNetVAE(resnet_3d, z_dim_3d)

        self.use_optical_flow = use_optical_flow

        if use_resnet:
            resnet_2d = models.resnet18(weights=None)
            resnet_2d.conv1 = nn.Conv2d(in_channels=2, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)

            self.vae_resnet_2d = ResNetVAE(resnet_2d, latent_dim=z_dim_2d)
        else:
            self.vae_resnet_2d = VAE(latent_dim=z_dim_2d, channels=[8, 16, 32], use_blocks=False, image_h=64, image_w=64,
                                     kernel_sizes=[3, 3, 3], strides=[1, 1, 1], num_input_channels=2)
            self.vae_resnet_2d.forward = self.vae_resnet_2d.latent_forward

        self.fc = SimpleFCN(input_size=z_dim_2d + z_dim_3d,
                       hidden_size=(z_dim_2d + z_dim_3d) // 2)

        self.device = device

    def get_optical_flows(self, video):
        # optical_flows = []

        batch_size, num_frames, channels, height, width = video.shape

        video1 = video[:, :-1, :, :, :]  # Shape: (batch_size, num_frames-1, channels, height, width)
        video2 = video[:, 1:, :, :, :] # Shape: (batch_size, num_frames-1, channels, height, width)

        video1 = video1.reshape(-1, channels, height,
                                width)  # Shape: (batch_size*(num_frames-1), channels, height, width)
        video2 = video2.reshape(-1, channels, height,
                                width)  # Shape: (batch_size*(num_frames-1), channels, height, width)

        optical_flow = self.model_raft(video1, video2)

        optical_flow = optical_flow[-1].detach().cpu()

        optical_flow = optical_flow.reshape(batch_size, num_frames - 1, *optical_flow.shape[1:])

        # print(optical_flow.shape)

        # for i in range(num_frames - 1):
        #     optical_flow = self.model_raft(video[:, i, :, :, :], video[:, i + 1, :, :, :])
        #     optical_flows.append(optical_flow[-1].detach().cpu())

            # each output: (batch_size, 2, h, w)

        # optical_flow = torch.stack(optical_flow, dim=1)
        optical_flow = optical_flow.permute(0, 2, 1, 3, 4)

        return optical_flow

    def get_latent_representation(self, video, spectrograms):

        if self.use_optical_flow:
            optical_flows = self.get_optical_flows(video).to(self.device)

            z_3d, mu_3d, log_var_3d = self.vae_resnet_3d(optical_flows)
        else:
            video = video.permute(0, 2, 1, 3, 4)

            z_3d, mu_3d, log_var_3d = self.vae_resnet_3d(video)

        z_2d, mu_2d, log_var_2d = self.vae_resnet_2d(spectrograms)

        z = torch.cat((z_2d, z_3d), dim=1)

        return z, mu_2d, log_var_2d, mu_3d, log_var_3d

    def forward(self, video, spectrograms):
        z, mu_2d, log_var_2d, mu_3d, log_var_3d = self.get_latent_representation(video, spectrograms)

        return self.fc(z), mu_2d, log_var_2d, mu_3d, log_var_3d

def train_video(data_loader_train, data_loader_val, lr=1e-03, epochs=50, verbose=True, name=None, z_dim_2d=64, z_dim_3d=64, use_optical_flow=False, use_resnet=True):
    model = VideoModel(z_dim_2d, z_dim_3d, use_optical_flow=use_optical_flow, use_resnet=use_resnet).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)

    print(f'Training {name}')

    criterion = BCELoss()

    best_val_accuracy = 0

    train_losses, train_accuracies, val_losses, val_accuracies = [], [], [], []

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_samples = 0
        for idx, batch in enumerate(data_loader_train):
            optimiser.zero_grad()

            spectrograms = batch['sources'].to(device)
            video = batch['video'].to(device)
            y = batch['label'].to(device)

            y_hat, mu_2d, log_var_2d, mu_3d, log_var_3d = model(video, spectrograms)

            loss_recon = criterion(y_hat.squeeze(), y.squeeze())

            loss_kl_1 = -0.5 * torch.mean(1 + log_var_2d - mu_2d.pow(2) - log_var_2d.exp())
            loss_kl_2 = -0.5 * torch.mean(1 + log_var_3d - mu_3d.pow(2) - log_var_3d.exp())

            loss = loss_recon + loss_kl_1 + loss_kl_2

            loss.backward()
            optimiser.step()

            train_loss += loss.item()

            predicted = (y_hat.squeeze() >= 0.5).float()
            correct = (predicted == y).sum().item()

            if correct > len(y):
                print('error')

            train_correct += correct
            train_samples += y.size(0)

            # del spectrograms, video, y, y_hat, mu_2d, log_var_2d, mu_3d, log_var_3d, loss_kl_1, loss_kl_2, loss
            # torch.cuda.empty_cache()

        avg_train_loss = train_loss / len(data_loader_train.dataset)
        train_accuracy = train_correct / train_samples

        val_loss = 0
        model.eval()
        with torch.no_grad():
            val_samples = 0
            val_correct = 0
            for batch in data_loader_val:
                spectrograms = batch['sources'].to(device)
                video = batch['video'].to(device)
                y = batch['label'].to(device)

                y_hat, mu_2d, log_var_2d, mu_3d, log_var_3d = model(video, spectrograms)

                loss_recon = criterion(y_hat.squeeze(), y.squeeze())

                loss_kl_1 = -0.5 * torch.mean(1 + log_var_2d - mu_2d.pow(2) - log_var_2d.exp())
                loss_kl_2 = -0.5 * torch.mean(1 + log_var_3d - mu_3d.pow(2) - log_var_3d.exp())

                loss = loss_recon + loss_kl_1 + loss_kl_2

                val_loss += loss.item()

                predicted = (y_hat.squeeze() >= 0.5).float()
                correct = (predicted == y).sum().item()
                val_correct += correct
                val_samples += y.size(0)

                del spectrograms, video, y
                torch.cuda.empty_cache()

        avg_val_loss = val_loss / len(data_loader_val.dataset)
        val_accuracy = val_correct / val_samples

        if verbose:
            print(
                f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.4f} Acc: {(train_accuracy*100):.4f}%, Val Loss: {avg_val_loss:.4f}, Acc: {(val_accuracy*100):.4f}%")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model = model
#
            if name:
                torch.save(best_model.state_dict(), f'checkpoints/{name}.pth')
                export_hyperparameters_to_file_video(name, z_dim_2d, z_dim_3d, data_loader_train.dataset.video_h, data_loader_train.dataset.video_w, data_loader_train.dataset.image_h, data_loader_train.dataset.image_w, data_loader_train.dataset.normalise, data_loader_train.dataset.fps)
                print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Best model saved as {name} with val accuracy: {(val_accuracy*100):.4f}%")

        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)

        with open(f'results/{name}_train_losses.pkl', 'wb') as f:
            pickle.dump(train_losses, f)

        with open(f'results/{name}_train_accuracies.pkl', 'wb') as f:
            pickle.dump(train_accuracies, f)

        with open(f'results/{name}_val_losses.pkl', 'wb') as f:
            pickle.dump(val_losses, f)

        with open(f'results/{name}_val_accuracies.pkl', 'wb') as f:
            pickle.dump(val_accuracies, f)