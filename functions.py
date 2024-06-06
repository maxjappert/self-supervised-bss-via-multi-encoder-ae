import json
import math
import pickle
import random

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from sklearn.model_selection import train_test_split
from torch import optim, nn
from torch.optim.lr_scheduler import StepLR

from models.cnn_multi_enc_ae_2d import ConvolutionalAutoencoder
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

from models.cnn_multi_enc_multi_dec_ae_2d import LinearConvolutionalAutoencoder
from models.separation_loss import WeightSeparationLoss

from skimage.metrics import structural_similarity as ssim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CircleTriangleDataset(Dataset):
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
        return (x - np.min(x)) / (np.max(x) - np.min(x))


    def __getitem__(self, idx):
        x, c, t = self.data[idx]
        x, c, t = self.min_max(x), self.min_max(c), self.min_max(t)
        x = torch.tensor(x, dtype=torch.float32)
        c = torch.tensor(c, dtype=torch.float32)
        t = torch.tensor(t, dtype=torch.float32)

        return x.permute(2, 0, 1), c.permute(2, 0, 1), t.permute(2, 0, 1)


def get_model(input_channels=1, image_hw=64, channels=[24, 48, 96, 144], hidden=96,
                 num_encoders=2, norm_type='group_norm',
                 use_weight_norm=True):
    return ConvolutionalAutoencoder(input_channels, image_hw, channels, hidden, num_encoders, norm_type, use_weight_norm)


def get_linear_model(input_channels=1, image_hw=64, channels=[24, 48, 96, 144], hidden=96,
                 num_encoders=2, norm_type='group_norm',
                 use_weight_norm=True):
    return LinearConvolutionalAutoencoder(input_channels, image_hw, channels, hidden, num_encoders, norm_type, use_weight_norm)


def get_split_dataloaders(dataset, split_ratio=0.8, batch_size=1024, num_workers=12):
    # Assuming `dataset` is your PyTorch Dataset
    dataset_size = len(dataset)
    indices = list(range(dataset_size))

    # Split ratio, e.g., 80% train, 20% validation
    split_ratio = 0.8
    split = int(split_ratio * dataset_size)

    # Shuffle and split indices
    train_indices, val_indices = train_test_split(indices, train_size=split_ratio, random_state=42)

    # Create samplers
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    # Create DataLoaders
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler, num_workers=num_workers)

    return train_loader, val_loader


def visualise_circle_triangle_predictions(sample, circle, triangle, x_pred, x_i_preds: list, name='test'):
    fig = plt.figure(figsize=(6, 6))
    grid = ImageGrid(fig, 111,
                    nrows_ncols=(2, 4),
                    axes_pad=0.15,
                    )

    labels = ['Mixed', 'Circle', 'Triangle']
    images = [sample, circle, triangle, None, x_pred] + x_i_preds
    y_labels = ['True', 'Pred.']
    for i, (ax, im) in enumerate(zip(grid, images)):
        if i != 3:
            if i < len(labels):
                ax.set_title(labels[i])
            if i % 4 == 0:
                ax.set_ylabel(y_labels[(i)//4])
            if i+1 == len(images):
                ax.set_title('(Dead Enc.)', color='gray', fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(im, cmap='gray')

    plt.savefig(f'{name}.png')
    plt.show()


def get_total_loss(x, x_pred, z, model, recon_loss, sep_loss, z_decay, zero_lr, sep_lr, linear=False):
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


def export_hyperparameters_to_file(name, channels, hidden, num_encoders, norm_type, use_weight_norm):
    variables = {
        'name': name,
        'channels': channels,
        'hidden': hidden,
        'num_encoders': num_encoders,
        'norm_type': norm_type,
        'use_weight_norm': use_weight_norm
    }

    with open(f'{name}.json', 'w') as file:
        json.dump(variables, file)


def get_hyperparameters_from_file(filename):
    with open(filename, 'r') as file:
        loaded_variables = json.load(file)

    return loaded_variables


def train(dataset_trainval, batch_size=1024, channels=[24, 48, 96, 144, 196], hidden=96,
                 num_encoders=2, norm_type='group_norm',
                 use_weight_norm=True, dataset_split_ratio=0.8, sep_norm='L1', sep_lr=0.5, zero_lr=0.01, lr=1e-3, lr_step_size=50, lr_gamma=0.1, weight_decay=1e-5, z_decay=1e-2, max_epochs=100, name=None, verbose=True, visualise=False, linear=False, test_save_step=1):

    if linear:
        model = get_linear_model(channels=channels, hidden=hidden,
                 num_encoders=num_encoders, norm_type=norm_type,
                 use_weight_norm=use_weight_norm)
    else:
        model = get_model(channels=channels, hidden=hidden,
                 num_encoders=num_encoders, norm_type=norm_type,
                 use_weight_norm=use_weight_norm)
    model.to(device)

    train_loader, val_loader = get_split_dataloaders(dataset_trainval, batch_size=batch_size)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)

    recon_loss = nn.BCEWithLogitsLoss()
    sep_loss = WeightSeparationLoss(model.num_encoders, sep_norm)

    if name:
        export_hyperparameters_to_file(name, channels, hidden, num_encoders, norm_type, use_weight_norm)

    train_losses = []
    val_losses = []
    best_val_loss = math.inf
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
            print(f'Epoch {epoch + 1}/{max_epochs}, Train Loss: {train_loss:.4f}')

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data in val_loader:
                x, c, t = data
                x = x.to(device)
                x_pred, z = model(x)

                loss = get_total_loss(x, x_pred, z, model, recon_loss, sep_loss, z_decay, zero_lr, sep_lr, linear=linear)

                val_loss += loss.item() * x.size(0)

        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        if val_loss < best_val_loss and name:
            torch.save(model.state_dict(), f"{name}_best.pth")

        test_loss = test(model, dataset_trainval, visualise=visualise if epoch % test_save_step == 0 else False,
                         name=str(epoch+1), num_samples=1)

        if verbose:
            print(f'Epoch {epoch + 1}/{max_epochs}, Validation Loss: {val_loss:.4f}, Score: {np.round(test_loss, 4)}')

    if name:
        torch.save(model.state_dict(), f"{name}_final.pth")

    return model, train_losses, val_losses


def evaluate_separation_ability(approxs, gts):
    matrix = np.empty([len(gts), len(approxs)])

    for i, approx in enumerate(gts):
        for j, gt in enumerate(approxs):
            if i != j:
                matrix[i, j] = ssim(approx, gt, data_range=gt.max() - gt.min())
            else:
                matrix[i, j] = -1

    return sum(np.max(matrix, axis=0)) / len(approxs)


def visualise_linear(model: LinearConvolutionalAutoencoder, dataset_test, visualise=True, name='test', num_samples=100):
    og_flag = model.return_sum
    model.return_sum = False
    sample, circle, triangle = dataset_test[random.randint(0, len(dataset_test) - 1)]
    x = torch.tensor(sample, dtype=torch.float32).unsqueeze(0).to(device)
    x_preds, _ = model(x)

    for i in range(len(x_preds)):
        x_preds[i] = torch.sigmoid(x_preds[i]).squeeze().detach().cpu().numpy()

    visualise_circle_triangle_predictions(sample.squeeze(), circle.squeeze(), triangle.squeeze(), sum(x_preds), x_preds,
                                          name=name)

    model.return_sum = og_flag


def test(model, dataset_test, visualise=True, name='test', num_samples=100):
    total_prediction_accuracy = 0

    for i in range(num_samples):
        # Sample random value from test set
        sample, circle, triangle = dataset_test[random.randint(0, len(dataset_test)-1)]

        x = torch.tensor(sample, dtype=torch.float32).unsqueeze(0).to(device)
        x_pred, _ = model(x)
        x_pred = torch.sigmoid(x_pred).squeeze().detach().cpu().numpy()

        with torch.no_grad():
            z = model.encode(x)

            masked_zs = []

            assert model.num_encoders == len(z)

            for i in range(len(z)):
                masked_zs.append([])
                for j in range(len(z)):
                    masked_zs[i].append(z[j] if i == j else torch.zeros_like(z[j]))

            x_i_preds = []
            for i in range(len(z)):
                y_i_pred = model.decode(masked_zs[i])
                x_i_pred = torch.sigmoid(y_i_pred).squeeze().detach().cpu().numpy()
                x_i_preds.append(x_i_pred)

            if visualise and i == 0:
                visualise_circle_triangle_predictions(sample.squeeze(), circle.squeeze(), triangle.squeeze(), x_pred, x_i_preds, name=name)
                print(f'{name}.png saved')

        total_prediction_accuracy += evaluate_separation_ability(x_i_preds, [circle.squeeze().numpy(), triangle.squeeze().numpy(), np.zeros_like(triangle.squeeze().numpy())])

    return total_prediction_accuracy / num_samples
