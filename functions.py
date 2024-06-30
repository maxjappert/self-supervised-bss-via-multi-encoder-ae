import json
import math
import os
import pickle
import random
import sys
import traceback

from PIL import Image
from datetime import datetime

import torchvision.transforms as transforms


import numpy as np
import torch
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from sklearn.model_selection import train_test_split
from torch import optim, nn
from torch.optim.lr_scheduler import StepLR

from evaluation_metric_functions import compute_spectral_snr, compute_spectral_sdr
from models.cnn_multi_enc_ae_2d_spectrograms import ConvolutionalAutoencoder
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

from models.cnn_multi_enc_multi_dec_ae_2d import LinearConvolutionalAutoencoder
from models.separation_loss import WeightSeparationLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CircleTriangleDataset(Dataset):
    """
    Used for the original toy scenario.
    """
    def __init__(self, transform=None):
        """
        Args:
            pickle_file_path (string): Path to the pickle file with the dataset.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        with open('data/single_channel_nonlinear_mixing_tri_circ.pickle', 'rb') as f:
            self.data = pickle.load(f, encoding='latin1')

    def __len__(self):
        return len(self.data)

    def min_max(self, x):
        """
        Simple min-max normalisation.
        :param x: The unnormalised input.
        :return: The normalised output in [0, 1].
        """
        return (x - np.min(x)) / (np.max(x) - np.min(x))


    def __getitem__(self, idx):
        x, c, t = self.data[idx]
        x, c, t = self.min_max(x), self.min_max(c), self.min_max(t)
        x = torch.tensor(x, dtype=torch.float32)
        c = torch.tensor(c, dtype=torch.float32)
        t = torch.tensor(t, dtype=torch.float32)

        return x.permute(2, 0, 1), c.permute(2, 0, 1), t.permute(2, 0, 1)


class TwoSourcesDataset(Dataset):
    """
    For any dataset where the mixes have two sources in the format of the Slakh dataset.
    """
    def __init__(self, split: str, name='slakh_two_sources_preprocessed', normalisation='minmax'):
        """
        Initialise the two source dataset.
        :param split: "train"/"validation"/"test"
        :param name: Name of the dataset folder name within the data folder.
        """
        self.data_folder_names = []

        self.master_path = os.path.join('data', name, split)

        for data_folder in os.listdir(self.master_path):
            #if random.random() < 0.05:
            self.data_folder_names.append(data_folder)

        self.normalisation = normalisation

    def __len__(self):
        return len(self.data_folder_names)

    def row_min_max(self, row):
        """
        Simple min-max normalisation, to squash into range [0, 1].
        :param row: Row vector where the first entry is the mix and the following entries are the stems.
        :return: Normalised row.
        """

        normalised_row = []

        for i, x in enumerate(row):
            S_db_normalized = (x - np.min(x)) / ((np.max(x) - np.min(x)) + 1e-7)
            normalised_row.append(torch.tensor(S_db_normalized).unsqueeze(0))

        return normalised_row

    def row_z_score(self, row):
        """
        Z-score normalization, to standardize data.
        :param row: Row vector where the first entry is the mix and the following entries are the stems.
        :return: Normalised row.
        """

        normalized_row = []

        for i, x in enumerate(row):
            mean = np.mean(x)
            std = np.std(x) + 1e-7
            S_db_normalized = (x - mean) / std
            normalized_row.append(torch.tensor(S_db_normalized).unsqueeze(0))

        return normalized_row

    def __getitem__(self, idx):

        chunks_master = np.array(Image.open((os.path.join(self.master_path, self.data_folder_names[idx], 'mix.png'))).convert('L'), dtype=np.float32)
        row = [chunks_master]

        for stem in os.listdir(os.path.join(self.master_path, self.data_folder_names[idx], 'stems')):
            stem_path = os.path.join(self.master_path, self.data_folder_names[idx], 'stems', stem)
            row.append(np.array(Image.open(stem_path).convert('L'), dtype=np.float32))

        return self.row_z_score(row) if self.normalisation == 'z-score' else self.row_min_max(row)


def save_spectrogram_to_file(spectrogram, filename):
    """
    Saves a spectrogram image.
    :param spectrogram: 2D array constituting a spectrogram.
    :param filename: Name of file within the images folder.
    :return: None.
    """
    plt.imsave(f'images/{filename}', spectrogram, cmap='gray')


def model_factory(input_channels=1, image_height=64, image_width=64, channels=[24, 48, 96, 144], hidden=96,
                  num_encoders=2, norm_type='group_norm',
                  use_weight_norm=True, linear=False):
    """
    Factory returning a model with the specified parameters.
    :param input_channels: Input channels. Default is 1.
    :param image_height: Image height.
    :param image_width: Image height.
    :param channels: Array of channels.
    :param hidden: Latent space size.
    :param num_encoders: Number of encoders and also decoders in the linear case.
    :param norm_type: "batch_norm"/"group_norm"/"layer_norm"/"instance_norm"/"None"
    :param use_weight_norm: Boolean deciding if the last layer weight vector w should be reparametrised into a
    unit vector v and a magnitude scalar g, s.t. w = (v/||v||)*g.
    :param linear: Boolean deciding if each encoder should have a separate decoder (linear) or a shared decoder with
    sparse mixing loss (non-linear).
    :return: The model with the specified parameters.
    """
    return LinearConvolutionalAutoencoder(input_channels, image_height, image_width, channels, hidden, num_encoders, norm_type, use_weight_norm) if linear else ConvolutionalAutoencoder(input_channels, image_height, image_width, channels, hidden, num_encoders, norm_type, use_weight_norm)


def visualise_predictions(x_gt, x_i_gts, x_pred, x_i_preds: list, name='test'):
    """
    Save an image file containing a visualisation of the separation.
    :param x_gt: Mixed ground truth spectrogram.
    :param x_i_gts: Array of ground truth stem spectrograms.
    :param x_pred: Approximated mixed spectrogram.
    :param x_i_preds: Approximated stem spectrograms as array.
    :param name: Filename.
    :return: None.
    """

    assert len(x_i_preds) == len(x_i_gts)

    num_sources = len(x_i_preds)

    fig = plt.figure(figsize=(2*num_sources, 6))
    grid = ImageGrid(fig, 111,
                    nrows_ncols=(2, 1 + num_sources),
                    axes_pad=0.15,
                    )

    #labels = ['Mixed', 'Circle', 'Triangle']
    #images = [sample, circle, triangle, None, x_pred] + x_i_preds
    images = [x_gt] + x_i_gts + [x_pred] + x_i_preds
    y_labels = ['True', 'Pred.']
    for i, (ax, im) in enumerate(zip(grid, images)):
        #if i < num_sources + 1:
        #    ax.set_title(labels[i])
        #if i % 4 == 0:
        #    ax.set_ylabel(y_labels[(i)//4])
        #if i+1 == len(images):
        #    ax.set_title('(Dead Enc.)', color='gray', fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(im, cmap='gray')

    if not os.path.exists('images'):
        os.mkdir('images')

    plt.savefig(f'images/{name}.png')
    #plt.show()


def get_total_loss(x, x_pred, z, model, recon_loss, sep_loss, z_decay, zero_lr, sep_lr, linear=False):
    """
    Compute the total loss to be propagated during training.
    :param x: Ground truth input.
    :param x_pred: Reconstructed input.
    :param z: Latent space.
    :param model: The model in question.
    :param recon_loss: Reconstruction loss.
    :param sep_loss: Separation or sparse mixing loss.
    :param z_decay: Z-decay, regularisation parameter.
    :param zero_lr: Learning rate for zero reconstruction loss.
    :param sep_lr: Learning rate for separation loss.
    :param linear: Boolean.
    :return: Reconstruction loss + separation loss + zero reconstruction loss
    """

    # Reconstruction loss
    loss = recon_loss(x_pred, x)

    # Add L2 regularisation penalising complex solutions
    for z_i in z:
        loss += z_decay * torch.mean(z_i ** 2)

    # Add separation (i.e., sparse mixing) loss
    if not linear:
        loss += sep_loss(model.decoder) * sep_lr

    # Add zero loss
    z_zeros = [torch.zeros(x.shape[0], model.hidden // model.num_encoders, z[0].shape[-1], z[0].shape[-1]).to(
        device) for _ in range(model.num_encoders)]
    x_pred_zeros = model.decode(z_zeros, True)
    zero_recon_loss = recon_loss(x_pred_zeros, torch.zeros_like(x_pred_zeros))
    loss += zero_recon_loss * zero_lr

    return loss


def export_hyperparameters_to_file(name, channels, hidden, num_encoders, norm_type, use_weight_norm, linear):
    """
    Saves the passed hyperparameters to a json file.
    :return: None
    """
    variables = {
        'name': name,
        'channels': channels,
        'hidden': hidden,
        'num_encoders': num_encoders,
        'norm_type': norm_type,
        'use_weight_norm': use_weight_norm,
        'linear': linear
    }

    if not os.path.exists('hyperparameters'):
        os.mkdir('hyperparameters')

    with open(f'hyperparameters/{name}.json', 'w') as file:
        json.dump(variables, file)


def get_hyperparameters_from_file(filename):
    with open(filename, 'r') as file:
        loaded_variables = json.load(file)

    return loaded_variables


def train(dataset_train, dataset_val, batch_size=64, channels=[24, 48, 96, 144, 196], hidden=196,
                 num_encoders=2, norm_type='group_norm', image_height=64, image_width=64,
                 use_weight_norm=True, dataset_split_ratio=0.8, sep_norm='L1', sep_lr=0.5, zero_lr=0.01, lr=1e-3,
          lr_step_size=50, lr_gamma=0.1, weight_decay=1e-5, z_decay=1e-2, max_epochs=100, name=None, verbose=True,
          visualise=False, linear=False, test_save_step=1, num_workers=12):
    """
    Trains a model.
    :param dataset_train: Train dataset.
    :param dataset_val: Validation dataset.
    :param batch_size: Batch size.
    :param channels: List of channels.
    :param hidden: Latent space size.
    :param num_encoders: Number of encoders and in the linear case decoders.
    :param norm_type: "batch_norm"/"group_norm"/"layer_norm"/"instance_norm"/"None"
    :param image_height: Height in input.
    :param image_width: Width of input.
    :param use_weight_norm: Boolean deciding if the last layer weight vector w should be reparametrised into a
    unit vector v and a magnitude scalar g, s.t. w = (v/||v||)*g.
    :param dataset_split_ratio: Dataset split ratio in [0, 1].
    :param sep_norm: "L1"/"L2"
    :param sep_lr: Learning rate for the separation loss.
    :param zero_lr: Learning rate for the zero reconstruction loss.
    :param lr: Learning rate of the total loss.
    :param lr_step_size: Step size for learning rate reduction.
    :param lr_gamma: Degree of learning rate reduction.
    :param weight_decay:Regularisation term penalising complex weights.
    :param z_decay: Another regularisation term penalising complex weights.
    :param max_epochs: Maximum number of training epochs.
    :param name: Filename for saving .pth to file.
    :param verbose: Boolean determining if training information should be printed to console.
    :param visualise: Boolean determining if images of the training progress should be saved to file.
    :param linear: Boolean.
    :param test_save_step: After how many steps an image is saved if visualise==True.
    :param num_workers: Number of threads for data loading.
    :return: Trained model, list of training losses per epoch, list of validation losses per epoch.
    """

    model = model_factory(channels=channels, hidden=hidden,
                              num_encoders=num_encoders, norm_type=norm_type,
                              use_weight_norm=use_weight_norm, image_height=image_height, image_width=image_width, linear=linear)
    model.to(device)

    if name:
        export_hyperparameters_to_file(name, channels, hidden, num_encoders, norm_type, use_weight_norm, linear)

    if not os.path.exists('checkpoints'):
        os.mkdir('checkpoints')

    #train_loader, val_loader = get_split_dataloaders(dataset_trainval, batch_size=batch_size, num_workers=num_workers)

    train_loader = DataLoader(dataset_train, batch_size=batch_size, num_workers=num_workers)
    val_loader = DataLoader(dataset_val, batch_size=batch_size, num_workers=num_workers)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)

    recon_loss = nn.BCEWithLogitsLoss()
    sep_loss = WeightSeparationLoss(model.num_encoders, sep_norm)

    train_losses = []
    val_losses = []
    best_sdr = -math.inf
    for epoch in range(max_epochs):
        model.train()        
        train_loss = 0.0
        for data in train_loader:
            x = data[0].to(device)

            optimizer.zero_grad()

            # Forward pass
            x_pred, z = model(x)

            loss = get_total_loss(x, x_pred, z, model, recon_loss, sep_loss, z_decay, zero_lr, sep_lr, linear=linear)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0)

        scheduler.step()

        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        if verbose:
            # Get the current timestamp
            now = datetime.now()

            # Format the timestamp as a string
            timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")
            print(f'[{timestamp_str}]:  Epoch {epoch + 1}/{max_epochs}, Train Loss: {train_loss:.4f}')

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data in val_loader:
                #x, c, t = data
                x = data[0].to(device)
                x_pred, z = model(x)

                loss = get_total_loss(x, x_pred, z, model, recon_loss, sep_loss, z_decay, zero_lr, sep_lr, linear=linear)

                val_loss += loss.item() * x.size(0)

        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        sdr = test(model, dataset_val, visualise=visualise if epoch % test_save_step == 0 else False,
                         name=f'{str(epoch + 1)}_{name}', num_samples=10, linear=linear, metric_function=compute_spectral_sdr)

        snr = test(model, dataset_val, visualise=visualise if epoch % test_save_step == 0 else False,
                         name=f'{str(epoch + 1)}_{name}', num_samples=10, linear=linear, metric_function=compute_spectral_snr)

        #ssim = test(model, dataset_val, visualise=visualise if epoch % test_save_step == 0 else False,
        #                 name=f'{str(epoch + 1)}_{name}', num_samples=10, linear=linear, metric_function=compute_ssim)


        if sdr > best_sdr and name:
            torch.save(model.state_dict(), f"checkpoints/{name}_best.pth")
            best_sdr = sdr

        if verbose:
            print(f'Epoch {epoch + 1}/{max_epochs}, Validation Loss: {val_loss:.4f}')
            print(f'SDR: {sdr}')
            print(f'SNR: {snr}')
            #print(f'SSIM: {ssim}')

    if name:
        torch.save(model.state_dict(), f"checkpoints/{name}_final.pth")

    return model, train_losses, val_losses


def evaluate_separation_ability(ground_truths, approximations, metric_function):
    # Ensure inputs are numpy arrays
    ground_truths = np.array(ground_truths)
    approximations = np.array(approximations)

    num_truths = len(ground_truths)
    num_approximations = len(approximations)

    # Initialize a matrix to store MSE values
    metric_matrix = np.zeros((num_truths, num_approximations))

    # Compute MSE for each pair of ground truth and approximation
    for i in range(num_truths):
        for j in range(num_approximations):
            metric_matrix[i, j] = metric_function(ground_truths[i], approximations[j]) # np.mean((ground_truths[i] - approximations[j]) ** 2)

    if np.isnan(metric_matrix).any():
        print('Metric matrix contains nan')
        return 0

    # Replace infinity values with a very large number
    large_number = 1e10
    quality_matrix = np.where(np.isinf(metric_matrix), large_number, metric_matrix)

    # Find the best matching for approximations to ground truths
    from scipy.optimize import linear_sum_assignment

    try:
        # Replace infinity values with a very large number
        large_number = 1e10
        quality_matrix = np.where(np.isinf(metric_matrix), large_number, metric_matrix)

        # Invert the Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(quality_matrix.max() - quality_matrix)
    except Exception as e:
        print(e)
        traceback.print_exc()
        print(metric_matrix)
        print(quality_matrix)
        print(quality_matrix.max() - quality_matrix)
        sys.exit(0)

    # Compute the overall quality score
    total = metric_matrix[row_ind, col_ind].sum()
    mean_mse = total / num_truths

    return mean_mse


# TODO: Rewrite using get_(non_)linear_separation(...) function
def visualise_linear(model: LinearConvolutionalAutoencoder, dataset_test, visualise=True, name='test', num_samples=100):
    og_flag = model.return_sum
    model.return_sum = False
    print('here??')
    sample, circle, triangle = dataset_test[random.randint(0, len(dataset_test) - 1)]
    x = torch.tensor(sample, dtype=torch.float32).unsqueeze(0).to(device)
    x_preds, _ = model(x)

    for i in range(len(x_preds)):
        x_preds[i] = torch.sigmoid(x_preds[i]).squeeze().detach().cpu().numpy()

    visualise_predictions(sample.squeeze(), circle.squeeze(), triangle.squeeze(), sum(x_preds), x_preds,
                          name=name)

    model.return_sum = og_flag


def get_linear_separation(model, sample):
    with torch.no_grad():
        x = torch.tensor(sample[0], dtype=torch.float32).unsqueeze(0).to(device)

        x_i_preds = []

        for source_idx in range(model.num_encoders):
            y_i_pred, z = model.forward_single_encoder(x, source_idx)
            x_i_preds.append(torch.sigmoid(y_i_pred).squeeze().detach().cpu().numpy())

    return x_i_preds


def get_non_linear_separation(model, sample):
    with torch.no_grad():
        x = torch.tensor(sample[0], dtype=torch.float32).unsqueeze(0).to(device)
        z = model.encode(x)
        masked_zs = []

        for i in range(len(z)):
            masked_zs.append([])
            for j in range(len(z)):
                masked_zs[i].append(z[j] if i == j else torch.zeros_like(z[j]))

        x_i_preds = []
        for i in range(len(z)):
            y_i_pred = model.decode(masked_zs[i])
            x_i_pred = torch.sigmoid(y_i_pred).squeeze().detach().cpu().numpy()
            x_i_preds.append(x_i_pred)

        return x_i_preds


def test(model, dataset_val, visualise=True, name='test', num_samples=100, single_file=True, linear=False, metric_function=compute_spectral_snr, random_visualisation=False):
    metric_sum = 0

    model.eval()

    if not os.path.exists('images'):
        os.makedirs('images')

    for sample_index in range(num_samples):
        # Sample random value from test set

        if sample_index == 0 and not random_visualisation:
            sample = dataset_val[0]
        else:
            sample = dataset_val[random.randint(0, len(dataset_val) - 1)]

        x = torch.tensor(sample[0], dtype=torch.float32).unsqueeze(0).to(device)
        x_pred, _ = model(x)
        x_pred = torch.sigmoid(x_pred).squeeze().detach().cpu().numpy()

        #save_spectrogram_to_file(x_pred, f'{name}_mix.png')

        x_i_preds = get_linear_separation(model, sample) if linear else get_non_linear_separation(model, sample)

        if visualise and sample_index == 0:
            if single_file:
                visualise_predictions(sample[0].squeeze(), [x_i.squeeze() for x_i in sample[1:]], x_pred, x_i_preds, name=name)
                print(f'{name}.png saved')
            else:
                save_spectrogram_to_file(x_pred, f'{name}_mix.png')
                save_spectrogram_to_file(sample[0].squeeze(), f'{name}_mix_gt.png')
                for l, x_i_pred in enumerate(x_i_preds):
                    save_spectrogram_to_file(x_i_pred, f'{name}_{l}.png')
                    save_spectrogram_to_file(sample[l+1].squeeze(), f'{name}_{l}_gt.png')

        metric_sum += evaluate_separation_ability(x_i_preds, [x_i_gt.squeeze().numpy() for x_i_gt in sample[1:]], metric_function)

    return metric_sum / num_samples
