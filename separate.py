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


def g(stems):
    return torch.sum(torch.stack(stems, dim=0) * alpha if type(stems) is list else stems * alpha, dim=0)


device = torch.device('cuda')

name = 'musdb_small_newelbo'
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

dataloader_train = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=12)
dataloader_val = DataLoader(val_dataset, batch_size=512, shuffle=True, num_workers=12)

L = 10
T = 100
alpha = 1/k
delta = 2 * 1e-09

gt_xs = [val_dataset[i]['spectrogram'].view(-1) for i in range(k)]
m = g(gt_xs).to(device)

original_shape = (image_h, image_w)
n = image_h

hps = json.load(open(f'hyperparameters/{name}.json'))
vae = VAE(latent_dim=hps['hidden'], image_h=image_h, image_w=image_w, use_blocks=hps['use_blocks'], channels=hps['channels'], kernel_sizes=hps['kernel_sizes'], strides=hps['strides']).to(
    device)

vae.load_state_dict(torch.load(f'checkpoints/{name}.pth', map_location=device))

xs = [torch.rand(image_h * image_w, requires_grad=True) for _ in range(k)]
# xs = [vae.decode(torch.randn(hps['hidden']).unsqueeze(dim=0).to(device)).squeeze(dim=0) for _ in range(k)]
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
                       visualise=True
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
        temp = (eta_i / sigmas[i] ** 2) * torch.eye(len(m.squeeze())) * (m.squeeze() - g(xs)).float()
        xs = u - temp

    x_chain.append(xs)
    print(f'Appended {i+1}/{L}')

final_xs = x_chain[-1].view((k, image_h, image_w))

for i in range(k):
    save_image(final_xs[i], f'images/000_recon_stem_{i+1}.png')

save_image(g(final_xs), 'images/000_m_recon.png')

