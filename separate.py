import numpy as np
import torch
from torchvision.utils import save_image

from functions_prior import VAE, PriorDataset

image_h = 1024
image_w = 128


def g(stems):
    return torch.sum(torch.stack(stems, dim=0) if type(stems) is list else stems, dim=0)


device = torch.device('cpu')

val_dataset = PriorDataset('val', image_h=image_h, image_w=image_w, name='toy_dataset', num_stems=4)

gt_xs = [val_dataset[i]['spectrogram'] for i in range(4)]

L = 10
T = 100
k = 4
batch_size = 4
x = torch.rand(k, 1, image_h, image_w, requires_grad=True)
alpha = torch.ones(k) * (1.0/k)
delta = 0.1
m = g(gt_xs)
vae = VAE(latent_dim=448, image_h=1024, image_w=128, kernel_size=5, channels=[8, 16, 32, 64, 128, 256, 512]).to(
    device)

vae.load_state_dict(torch.load(f'checkpoints/optimal1_toy.pth', map_location=device))

sigma_start = 0.01
sigma_end = 1
sigmas = torch.logspace(start=torch.log10(torch.tensor(sigma_start)),
                        end=torch.log10(torch.tensor(sigma_end)),
                        steps=L).flip(dims=[0])

print(sigmas)

x_chain = []

for i in range(L):
    eta_i = delta * sigmas[i]**2 / sigmas[L-1]**2

    for t in range(T):
        epsilon_t = torch.randn(x.shape)
        elbo = vae.log_prob(x.float()).mean(axis=0).squeeze()
        grad_log_p_x = torch.autograd.grad(elbo, x, retain_graph=True)[0]
        u = x + eta_i * grad_log_p_x + torch.sqrt(2*eta_i) * epsilon_t
        x = u - (eta_i / sigmas[i]**2) * (m - g(x))

    x_chain.append(x)
    print(f'Appended {i+1}/{L}')

final_x = x_chain[-1]

for i in range(4):
    save_image(final_x[i], f'images/recon_stem_{i+1}.png')

