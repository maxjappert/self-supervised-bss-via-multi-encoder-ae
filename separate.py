import json

import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision.utils import save_image

from functions_prior import VAE, PriorDataset



def g(stems):
    return torch.sum(torch.stack(stems, dim=0) if type(stems) is list else stems, dim=0)


device = torch.device('cpu')

name = 'cyclic_musdb'
image_h = 1024

if name.__contains__('toy'):
    image_w = 128
    val_dataset = PriorDataset('val', image_h=image_h, image_w=image_w, name='toy_dataset', num_stems=4)
else:
    image_w = 384
    val_dataset = PriorDataset('val', image_h=image_h, image_w=image_w, name='musdb_18_prior', num_stems=4)

gt_xs = [val_dataset[i]['spectrogram'] for i in range(4)]

L = 10
T = 100
k = 4
batch_size = 4
xs = [torch.rand(image_h, image_w, requires_grad=True) for _ in range(4)]
alpha = torch.ones(k) * (1.0/k)
delta = 0.1
m = g(gt_xs)

hps = json.load(open(f'hyperparameters/{name}.json'))
vae = VAE(latent_dim=hps['hidden'], image_h=image_h, image_w=image_w, kernel_size=hps['kernel_size'], channels=hps['channels']).to(
    device)

vae.load_state_dict(torch.load(f'checkpoints/{name}.pth', map_location=device))

sigma_start = 0.01
sigma_end = 1
sigmas = torch.logspace(start=torch.log10(torch.tensor(sigma_start)),
                        end=torch.log10(torch.tensor(sigma_end)),
                        steps=L).flip(dims=[0])

# plt.imshow(m.squeeze(), cmap='gray')
# plt.show()

# for l in range(4):
#     plt.imshow(gt_xs[l].detach(), cmap='grey')
#     plt.show()

x_chain = []

for i in range(L):
    eta_i = delta * sigmas[i]**2 / sigmas[L-1]**2

    for t in range(T):
        epsilon_t = torch.randn((image_h, image_w))
        new_xs = []
        for x in xs:
            elbo = vae.log_prob(x.unsqueeze(dim=0).unsqueeze(dim=0).float()).squeeze()
            grad_log_p_x = torch.autograd.grad(elbo, x, retain_graph=True)[0]
            u = x + eta_i * grad_log_p_x + torch.sqrt(2 * eta_i) * epsilon_t
            new_x = u - (eta_i / sigmas[i] ** 2) * (m - g(xs))
            new_xs.append(x)

        xs = new_xs

    for l in range(4):
        plt.imshow(xs[l].detach(), cmap='grey')
        plt.show()

    x_chain.append(xs)
    print(f'Appended {i+1}/{L}')

final_x = x_chain[-1]

for i in range(4):
    save_image(final_x[i], f'images/recon_stem_{i+1}.png')

