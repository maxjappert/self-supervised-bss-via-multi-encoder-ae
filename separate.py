import json
import sys

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import optim
from torchvision.utils import save_image

from functions_prior import VAE, PriorDataset



def g(stems):
    return torch.sum(torch.stack(stems, dim=0) * alpha if type(stems) is list else stems * alpha, dim=0)


device = torch.device('cuda')

name = 'musdb_small_newelbo'
hps = json.load(open(f'hyperparameters/{name}.json'))
image_h = hps['image_h']
image_w = hps['image_w']

if name.__contains__('toy'):
    val_dataset = PriorDataset('val', image_h=image_h, image_w=image_w, name='toy_dataset', num_stems=4)
else:
    val_dataset = PriorDataset('val', image_h=image_h, image_w=image_w, name='musdb_18_prior', num_stems=4)

gt_xs = [val_dataset[i]['spectrogram'] for i in range(4)]

L = 10
T = 100
k = 4
batch_size = 4
alpha = (1.0/k)
delta = 2 * 1e-06
m = g(gt_xs).to(device)

n = m.shape[1]

hps = json.load(open(f'hyperparameters/{name}.json'))
vae = VAE(latent_dim=hps['hidden'], image_h=image_h, image_w=image_w, use_blocks=hps['use_blocks'], channels=hps['channels'], kernel_sizes=hps['kernel_sizes'], strides=hps['strides']).to(
    device)

vae.load_state_dict(torch.load(f'checkpoints/{name}.pth', map_location=device))

xs = [torch.rand(1, image_h, image_w, requires_grad=True) for _ in range(4)]
# xs = [vae.decode(torch.randn(hps['hidden']).unsqueeze(dim=0).to(device)).squeeze(dim=0) for _ in range(4)]
xs = torch.stack(xs, dim=0).to(device)

for i, x in enumerate(gt_xs):
    save_image(x, f'images/000_gt_stem_{i}.png')

sigma_start = 0.01
sigma_end = 1
sigmas = torch.logspace(start=torch.log10(torch.tensor(sigma_start)),
                        end=torch.log10(torch.tensor(sigma_end)),
                        steps=L).flip(dims=[0]).to(device)
sigmas.requires_grad_(True)

save_image(m.squeeze(), 'images/000_m.png')

x_chain = []

for i in range(L):
    eta_i = delta * sigmas[i]**2 / sigmas[L-1]**2

    for t in range(T):
        epsilon_t = torch.randn(xs.shape, requires_grad=True).to(device)
        new_xs = []

        elbo = vae.log_prob(xs.float().to(device))
        grad_log_p_x = torch.autograd.grad(elbo, xs, retain_graph=True)[0]
        u = xs + eta_i * grad_log_p_x + torch.sqrt(2 * eta_i) * epsilon_t
        temp = (eta_i / sigmas[i] ** 2) * (torch.eye(n) * alpha).to(device) @ (m.squeeze() - g(xs).squeeze()).float()
        xs = u - temp

    x_chain.append(xs)
    print(f'Appended {i+1}/{L}')

final_xs = x_chain[-1]

for i in range(4):
    save_image(final_xs[i], f'images/000_recon_stem_{i+1}.png')

save_image(g(final_xs), 'images/000_m_recon.png')

