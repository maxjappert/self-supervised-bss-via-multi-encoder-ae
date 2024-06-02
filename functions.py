import pickle

import numpy as np
import torch
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from sklearn.model_selection import train_test_split
from torch import optim, nn
from torch.optim.lr_scheduler import StepLR

from models.cnn_multi_enc_ae_2d import ConvolutionalAutoencoder
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

from models.separation_loss import WeightSeparationLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


def get_split_dataloaders(dataset, split_ratio=0.8, batch_size=512, num_workers=12):
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
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)

    return train_loader, val_loader


def visualise_circle_triangle_predictions(sample, circle, triangle, x_pred, x_i_preds: list):
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

    plt.savefig('test.png')
    plt.show()

def get_total_loss(x, x_pred, z, model, recon_loss, sep_loss, z_decay, zero_lr, sep_lr):
    # Reconstruction loss
    loss = recon_loss(x_pred, x)

    # Add L2 regularisation penalising complex solutions
    for z_i in z:
        loss += z_decay * torch.mean(z_i ** 2)

    # Add separation (i.e., sparse mixing) loss
    loss += sep_loss(model.decoder) * sep_lr

    # Add zero loss
    z_zeros = [torch.zeros(x.shape[0], model.hidden // model.num_encoders, z[0].shape[-1], z[0].shape[-1]).to(
        device) for _ in range(model.num_encoders)]
    x_pred_zeros = model.decode(z_zeros, True)
    zero_recon_loss = recon_loss(x_pred_zeros, torch.zeros_like(x_pred_zeros))
    loss += zero_recon_loss * zero_lr

    return loss


def train(model: ConvolutionalAutoencoder, dataset_trainval, batch_size=512, dataset_split_ratio=0.8, sep_norm='L1', sep_lr=0.5, zero_lr=0.01, lr=1e-3, lr_step_size=50, lr_gamma=0.1, weight_decay=1e-5, z_decay=1e-2, max_epochs=300, name=None):
    model.to(device)

    train_loader, val_loader = get_split_dataloaders(dataset_trainval)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)

    recon_loss = nn.BCEWithLogitsLoss()
    sep_loss = WeightSeparationLoss(model.num_encoders, sep_norm)

    for epoch in range(max_epochs):
        model.train()
        train_loss = 0.0
        for data in train_loader:
            x = data[0].to(device)

            optimizer.zero_grad()

            # Forward pass
            x_pred, z = model(x)

            loss = get_total_loss(x, x_pred, z, model, recon_loss, sep_loss, z_decay, zero_lr, sep_lr)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0)

        scheduler.step()

        train_loss /= len(train_loader.dataset)
        print(f'Epoch {epoch + 1}/{max_epochs}, Train Loss: {train_loss:.4f}')

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data in val_loader:
                x, c, t = data
                x = x.to(device)
                x_pred, z = model(x)

                loss = get_total_loss(x, x_pred, z, model, recon_loss, sep_loss, z_decay, zero_lr, sep_lr)

                val_loss += loss.item() * x.size(0)

        val_loss /= len(val_loader.dataset)
        print(f'Epoch {epoch + 1}/{max_epochs}, Validation Loss: {val_loss:.4f}')

    if name:
        torch.save(model.state_dict(), f"{name}.pth")

    return model


def test(model: ConvolutionalAutoencoder, dataset_test):
    sample, circle, triangle = dataset_test[0]

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

        visualise_circle_triangle_predictions(sample.squeeze(), circle.squeeze(), triangle.squeeze(), x_pred, x_i_preds)
