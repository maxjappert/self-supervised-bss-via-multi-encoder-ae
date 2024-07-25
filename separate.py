import json
import sys

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import optim
from torch.nn import MSELoss
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from functions_prior import VAE, PriorDataset, finetune_sigma, train_vae


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
        new_tensor[i] = minmax_normalise(old_tensor[i])

    return new_tensor




def g(stems):
    return minmax_normalise(torch.sum(torch.stack(stems, dim=0) * alpha if type(stems) is list else stems * alpha, dim=0))


device = torch.device('cuda')

# name = 'musdb_small_newelbo'
name = 'musdb_tiny_optimal_2'
hps = json.load(open(f'hyperparameters/{name}.json'))
image_h = hps['image_h']
image_w = hps['image_w']

k = 2

debug = True

if name.__contains__('toy'):
    train_dataset = PriorDataset('train', image_h=image_h, image_w=image_w, name='toy_dataset', num_stems=k, debug=debug)
    val_dataset = PriorDataset('val', image_h=image_h, image_w=image_w, name='toy_dataset', num_stems=k, debug=debug)
else:
    train_dataset = PriorDataset('train', image_h=image_h, image_w=image_w, name='musdb_18_prior', num_stems=k, debug=debug)
    val_dataset = PriorDataset('val', image_h=image_h, image_w=image_w, name='musdb_18_prior', num_stems=k, debug=debug)

dataloader_train = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=12)
dataloader_val = DataLoader(val_dataset, batch_size=256, shuffle=True, num_workers=12)

L = 10
T = 100
alpha = 1# /k
delta = 2 * 1e-05

gt_xs = [val_dataset[i]['spectrogram'].view(-1) for i in range(k)]
m = g(gt_xs).to(device)

original_shape = (image_h, image_w)
n = image_h

hps = json.load(open(f'hyperparameters/{name}.json'))
vae = VAE(latent_dim=hps['hidden'], image_h=image_h, image_w=image_w, use_blocks=hps['use_blocks'], channels=hps['channels'], kernel_sizes=hps['kernel_sizes'], strides=hps['strides']).to(
    device)

vae.load_state_dict(torch.load(f'checkpoints/{name}.pth', map_location=device))

xs = [torch.rand(image_h * image_w, requires_grad=True) for _ in range(k)]
# xs = [vae.decode(torch.randn(hps['hidden']).unsqueeze(dim=0).to(device)).squeeze(dim=0).view(-1) for _ in range(k)]
xs = torch.stack(xs, dim=0).to(device)

for i, x in enumerate(gt_xs):
    save_image(x.view(original_shape), f'images/000_gt_stem_{i}.png')

sigma_start = 0.1
sigma_end = 0.5
sigmas = torch.logspace(start=torch.log10(torch.tensor(sigma_start)),
                        end=torch.log10(torch.tensor(sigma_end)),
                        steps=L, base=10).flip(dims=[0])
sigmas.requires_grad_(True)

save_image(m.view(original_shape), 'images/000_m.png')

x_chain = []

def finetune_sigma_models():
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

def train_sigma_models():
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

finetune_sigma_models()

for i in range(L):
    eta_i = delta * sigmas[i]**2 / sigmas[L-1]**2

    name = f'sigma_{np.round(sigmas[i].detach().item(), 3)}'
    vae = VAE(latent_dim=hps['hidden'], image_h=image_h, image_w=image_w, use_blocks=hps['use_blocks'],
              channels=hps['channels'], kernel_sizes=hps['kernel_sizes'], strides=hps['strides']).to(
        device)
    vae.load_state_dict(torch.load(f'checkpoints/{name}.pth', map_location=device))

    for t in range(T):
        epsilon_t = torch.randn(xs.shape, requires_grad=True).to(device)

        elbo = vae.log_prob(xs.view((k, 1, image_h, image_w))).float().to(device)
        grad_log_p_x = torch.autograd.grad(elbo, xs, retain_graph=True)[0]
        u = xs + eta_i * grad_log_p_x + torch.sqrt(2 * eta_i) * epsilon_t
        temp = (eta_i / sigmas[i] ** 2) * (m.squeeze() - g(xs)).float()
        xs = u - temp
        #xs[0] = minmax_normalise(xs[0])
        #xs[1] = minmax_normalise(xs[1])
        xs = minmax_rows(xs)

    x_chain.append(xs)

    # for j in range(k):
    #     save_image(x_chain[-1][j].view(image_h, image_w), f'images/000_recon_stem_{j + 1}.png')

    print(f'Appended {i+1}/{L}')

final_xs = x_chain[-1].view((k, image_h, image_w))

for i in range(k):
    save_image(final_xs[i], f'images/000_recon_stem_{i+1}.png')

m_recon = g(final_xs)
save_image(m_recon, 'images/000_m_recon.png')

