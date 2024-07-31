import json
import random
import sys

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import optim
from torch.nn import MSELoss
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from functions_prior import VAE, PriorDataset, finetune_sigma, train_vae

def extract_x(xz, stem_idx):
    start_idx = stem_idx * (x_dim + z_dim)
    end_idx = start_idx + x_dim
    return xz[start_idx:end_idx]


def extract_z(xz, stem_idx):
    start_idx = (stem_idx+1) * x_dim + stem_idx * z_dim
    end_idx = start_idx + z_dim
    return xz[start_idx:end_idx]


def minmax_normalise(tensor, min_value=0.0, max_value=1.0):
    """
    Apply Min-Max normalisation to a tensor.

    Args:
    - tensor (torch.Tensor): The input tensor to be normalised.
    - min_value (float): The minimum value of the desired range.
    - max_value (float): The maximum value of the desired range.

    Returns:
    - torch.Tensor: The normalised tensor.
    """
    tensor_min = tensor.min()
    tensor_max = tensor.max()
    norm_tensor = (tensor - tensor_min) / (tensor_max - tensor_min)
    norm_tensor = norm_tensor * (max_value - min_value) + min_value
    return norm_tensor

def minmax_rows(old_tensor):
    new_tensor = torch.zeros_like(old_tensor)

    for i in range(old_tensor.shape[0]):
        individual_datapoint = old_tensor[i]
        new_tensor[i] = minmax_normalise(individual_datapoint)

    return new_tensor

torch.autograd.set_detect_anomaly(True)

def gradient_log_px(x, vae):
    x = x.clone().detach().requires_grad_(True)  # Clone and enable gradient computation with respect to x

    if x.grad is not None:
        x.grad.zero_()  # Clear existing gradients if any

    # x.requires_grad_(True)  # Enable gradient computation with respect to x
    elbo = vae.log_prob(x)
    elbo.backward()  # Compute gradients
    # grad_log_px = x.grad  # Get the gradient of the ELBO with respect to x
    # x.requires_grad_(False)  # Disable gradient computation
    grad_log_px = x.grad.clone().detach()  # Clone the gradient to avoid in-place modifications
    return grad_log_px


def g(stems):
    return torch.sum(torch.stack(stems, dim=0) * alpha if type(stems) is list else stems * alpha, dim=0)


def g_xz(xz):
    total_sum = 0

    for stem_idx in range(k):
        total_sum += torch.cat([extract_x(xz, stem_idx).to(device), torch.zeros(z_dim).to(device)])

    return total_sum


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# name = 'musdb_small_newelbo'
name = 'toy'
hps = json.load(open(f'hyperparameters/{name}.json'))
image_h = hps['image_h']
image_w = hps['image_w']

k = 2

debug = False

train_datasets = [PriorDataset('train', image_h=image_h, image_w=image_w, name='toy_dataset', num_stems=4, debug=debug, stem_type=i+1) for i in range(4)]
val_datasets = [PriorDataset('val', image_h=image_h, image_w=image_w, name='toy_dataset', num_stems=4, debug=debug, stem_type=i+1) for i in range(4)]

dataloaders_train = [DataLoader(train_datasets[i], batch_size=256, shuffle=True, num_workers=12) for i in range(4)]
dataloaders_val = [DataLoader(val_datasets[i], batch_size=256, shuffle=True, num_workers=12) for i in range(4)]

L = 10
T = 100
alpha = 1
delta = 2 * 1e-05

original_shape = (image_h, image_w)
x_dim = image_h*image_w
z_dim = hps['hidden']

# gt_xs = [val_datasets[i][1]['spectrogram'] for i in range(k)]
# tODO
gt_xs = [val_datasets[0][1]['spectrogram'], val_datasets[3][1]['spectrogram']]
m = g(gt_xs).to(device)

gt_xz = [torch.cat([gt_xs[i].view(-1), torch.zeros(z_dim)]) for i in range(k)]
gt_xz = torch.cat(gt_xz)
m_xz = g_xz(gt_xz).to(device)

hps = json.load(open(f'hyperparameters/{name}.json'))

vaes = []

# todo
for i, stem_type in enumerate([0, 3]):
    vaes.append(VAE(latent_dim=hps['hidden'],
                    image_h=image_h,
                    image_w=image_w,
                    use_blocks=hps['use_blocks'],
                    channels=hps['channels'],
                    kernel_sizes=hps['kernel_sizes'],
                    strides=hps['strides']).to(device))

    vae_name = f'{name}_stem{stem_type+1}'
    vaes[i].load_state_dict(torch.load(f'checkpoints/{vae_name}.pth', map_location=device))

xz = []

for i in range(k):
    noise_image = torch.rand(image_h, image_w).to(device)
    mu, log_var = vaes[i].encode(noise_image.unsqueeze(dim=0).unsqueeze(dim=0))
    z = vaes[i].reparameterise(mu, log_var)
    xz.append(noise_image.view(-1))
    xz.append(z.view(-1))

# create a big flat vector
xz = torch.cat(xz).to(device)
xz.requires_grad_(True)
assert len(xz) == k * (x_dim + z_dim)

for i, x in enumerate(gt_xs):
    save_image(x, f'images/000_gt_stem_{i}_new.png')

sigma_start = 0.1
sigma_end = 0.5
sigmas = torch.logspace(start=torch.log10(torch.tensor(sigma_start)),
                        end=torch.log10(torch.tensor(sigma_end)),
                        steps=L, base=10).flip(dims=[0])
sigmas.requires_grad_(True)

save_image(m, 'images/000_m.png')

xz_chain = []

def finetune_sigma_models(vae, dataloader_train, dataloader_val):
    for sigma in sigmas:
        finetune_sigma(vae,
                       dataloader_train,
                       dataloader_val,
                       sigma=sigma.detach().item(),
                       verbose=True,
                       visualise=True,
                       lr=0.001,
                       epochs=20
                       )

def train_sigma_models(dataloader_train, dataloader_val):
    for sigma in sigmas:
        train_vae(dataloader_train,
                  dataloader_val,
                  sigma=sigma.detach().item(),
                  lr=1e-03,
                  cyclic_lr=False,
                  kernel_sizes=[3, 3, 3, 3, 3],
                  strides=[1, 2, 2, 2, 1],
                  channels=[32, 64, 128, 256, 512],
                  name=f'sigma_{np.round(sigma.detach().item(), 3)}',
                  criterion=MSELoss(),
                  epochs=40,
                  contrastive_loss=False,
                  recon_weight=1,
                  use_blocks=False,
                  latent_dim=512,
                  kld_weight=1,
                  visualise=True,
                  image_h=image_h,
                  image_w=image_w
                  )

# finetune_sigma_models()

def log_p_z(xz):
    total_log_prob = 0
    mvsn = torch.distributions.normal.Normal(0, 1)

    for stem_idx in range(k):
        z = extract_z(xz, stem_idx)

        assert len(z) == z_dim

        total_log_prob += torch.sum(mvsn.log_prob(z))


    return total_log_prob


def log_p_x_given_z(vaes, xz, sigma):
    total_log_prob = 0

    for stem_idx in range(k):

        x = extract_x(xz, stem_idx)
        z = extract_z(xz, stem_idx)

        x_recon = vaes[stem_idx].decode(z.unsqueeze(dim=0))
        mvn = torch.distributions.normal.Normal(x_recon.view(-1), sigma.to(device)**2)

        total_log_prob += torch.sum(mvn.log_prob(x))

    return total_log_prob


for i in range(L):
    eta_i = delta * sigmas[i]**2 / sigmas[L-1]**2

    # name = f'sigma_{np.round(sigmas[i].detach().item(), 3)}'
    # vae = VAE(latent_dim=hps['hidden'], image_h=image_h, image_w=image_w, use_blocks=hps['use_blocks'],
    #           channels=hps['channels'], kernel_sizes=hps['kernel_sizes'], strides=hps['strides']).to(
    #     device)
    # vae.load_state_dict(torch.load(f'checkpoints/{name}.pth', map_location=device))

    for t in range(T):

        epsilon_t = torch.randn(xz.shape, requires_grad=True).to(device)

        if xz.grad is not None:
            xz.grad.zero_()

        # elbo = vae.log_prob(xs[source_idx].unsqueeze(dim=0)).to(device)
        # grad_log_p_x = #torch.autograd.grad(elbo, xs[source_idx], retain_graph=True, create_graph=True)[0]
        log_p_x_z = log_p_z(xz) + log_p_x_given_z(vaes, xz, sigmas[i])

        # TODO: Why allow_unused?
        grad_log_p_x_z = torch.autograd.grad(log_p_x_z, xz, retain_graph=True, create_graph=True)[0]

        u = xz + eta_i * grad_log_p_x_z + torch.sqrt(2 * eta_i) * epsilon_t
        constraint_term = (eta_i / sigmas[i] ** 2) * (m_xz - g_xz(xz)).float() * alpha
        elongated_constraint_term = [constraint_term for _ in range(k)]
        xz = u - torch.cat(elongated_constraint_term)# minmax_rows(u - temp)

    # plt.imshow(xs[0].squeeze().detach(), cmap='gray')
    # plt.show()
    # plt.imshow(xs[1].squeeze().detach(), cmap='gray')
    # plt.show()

    # x_chain.append((xs-1)*-1)
    xz_chain.append(xz)
    # x_chain.append((minmax_rows(xs)-1)*-1)

    # for j in range(k):
    #     save_image(x_chain[-1][j].view(image_h, image_w), f'images/000_recon_stem_{j + 1}.png')

    print(f'Appended {i+1}/{L}')

final_xz = xz_chain[-1]
final_xs = []

for i in range(k):
    # plt.imshow(final_xs[i].squeeze().detach(), cmap='gray')
    # plt.show()
    x = extract_x(final_xz, i).view(image_h, image_w)
    final_xs.append(x)
    save_image(x, f'images/000_recon_stem_{i+1}_new.png')

m_recon = g(final_xs)
save_image(m_recon, 'images/000_m_recon_new.png')

