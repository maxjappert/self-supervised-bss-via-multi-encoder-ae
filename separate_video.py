import math
from datetime import datetime
import json
import random
import sys
from inspect import stack

import mir_eval
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import optim
from torch.nn import MSELoss
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os

from functions_prior import VAE, PriorDataset, finetune_sigma, train_vae, VideoModel
from separate_new import get_vaes_rochester

k = 2

def extract_x(xz, stem_idx, x_dim, z_dim):
    start_idx = stem_idx * (x_dim + z_dim)
    end_idx = start_idx + x_dim
    return xz[start_idx:end_idx]


def extract_z(xz, stem_idx, x_dim, z_dim):
    start_idx = (stem_idx + 1) * x_dim + stem_idx * z_dim
    end_idx = start_idx + z_dim
    return xz[start_idx:end_idx]

def extract_stacked_x(xz, x_dim, z_dim):
    xs = []
    for i in range(k):
        x = extract_x(xz, i, x_dim, z_dim)
        x = torch.tensor(x)
        xs.append(x)

    return torch.stack(xs).view(1, 2, int(math.sqrt(x_dim)), int(math.sqrt(x_dim)))


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


def g(stems, alpha=1):
    return torch.sum(torch.stack(stems, dim=0) * alpha if type(stems) is list else stems * alpha, dim=0)


def g_xz(xz, x_dim, z_dim, device):
    total_sum = 0

    for stem_idx in range(k):
        total_sum += torch.cat([extract_x(xz, stem_idx, x_dim, z_dim).to(device), torch.zeros(z_dim).to(device)])

    return total_sum


def finetune_sigma_models(vae, dataloader_train, dataloader_val, sigmas, name):
    for sigma in sigmas:
        finetune_sigma(vae,
                       dataloader_train,
                       dataloader_val,
                       sigma=sigma.detach().item(),
                       verbose=True,
                       visualise=True,
                       lr=0.001,
                       epochs=30,
                       parent_name=name
                       )


image_h = 64
image_w = 64


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


if len(sys.argv) > 1:
    train = sys.argv[1] == 'train'
else:
    train = False


if train:
    name = 'toy'

    device = 'cuda'

    debug = False

    hps = json.load(open(f'hyperparameters/{name}.json'))

    vae = VAE(latent_dim=hps['hidden'],
              image_h=image_h,
              image_w=image_w,
              use_blocks=hps['use_blocks'],
              channels=hps['channels'],
              kernel_sizes=hps['kernel_sizes'],
              strides=hps['strides']).to(device)

    stem_type = int(sys.argv[2])

    assert stem_type in range(1, 5)

    train_dataset = PriorDataset('train', image_h=image_h, image_w=image_w, name='toy_dataset', num_stems=4,
                                 debug=debug,
                                 stem_type=stem_type)

    val_dataset = PriorDataset('val', image_h=image_h, image_w=image_w, name='toy_dataset', num_stems=4, debug=debug,
                               stem_type=stem_type)

    dataloader_train = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4)
    dataloader_val = DataLoader(val_dataset, batch_size=256, shuffle=True, num_workers=4)

    vae_name = f'{name}_stem{stem_type}'
    vae.load_state_dict(torch.load(f'checkpoints/{vae_name}.pth', map_location=device))

    sigma_start = 0.1
    sigma_end = 1.0
    L = 10
    sigmas = torch.logspace(start=torch.log10(torch.tensor(sigma_start)),
                            end=torch.log10(torch.tensor(sigma_end)),
                            steps=L, base=10).flip(dims=[0])

    finetune_sigma_models(vae, dataloader_train, dataloader_val, sigmas)

    sys.exit(0)


def get_vaes(name, stem_indices, sigma=None):
    hps = json.load(open(os.path.join('hyperparameters', f'{name}_stem1.json')))
    vaes = []

    for stem_index in stem_indices:
        vae = VAE(latent_dim=hps['hidden'],
                          image_h=image_h,
                          image_w=image_w,
                          use_blocks=hps['use_blocks'],
                          channels=hps['channels'],
                          kernel_sizes=hps['kernel_sizes'],
                          strides=hps['strides']).to(device)

        vae_name = f'{name}_stem{stem_index + 1}' if sigma is None else f'sigma_{name}_stem{stem_index + 1}_{sigma}'

        # vae_name = f'{name}_stem{stem_type + 1}'
        vae.load_state_dict(torch.load(f'checkpoints/{vae_name}.pth', map_location=device))

        vaes.append(vae)

    return vaes


def log_p_z(xz, x_dim, z_dim, k):
    total_log_prob = 0
    mvsn = torch.distributions.normal.Normal(0, 1)

    for stem_idx in range(k):
        z = extract_z(xz, stem_idx, x_dim=x_dim, z_dim=z_dim)

        assert len(z) == z_dim

        total_log_prob += torch.sum(mvsn.log_prob(z))

    return total_log_prob


def log_p_x_given_z(vaes, xz, sigma, x_dim, z_dim, k, device):
    total_log_prob = 0

    for stem_idx in range(k):
        x = extract_x(xz, stem_idx, x_dim=x_dim, z_dim=z_dim)
        z = extract_z(xz, stem_idx, x_dim=x_dim, z_dim=z_dim)

        x_recon = vaes[stem_idx].decode(z.unsqueeze(dim=0).to(device))
        mvn = torch.distributions.normal.Normal(x_recon.view(-1), sigma)

        total_log_prob += torch.sum(mvn.log_prob(x))

        del x_recon, mvn, x, z

    return total_log_prob

def log_p_s(model, video, xz, x_dim, z_dim, k, device):
    x_2c = [extract_x(xz, stem_idx, x_dim=x_dim, z_dim=z_dim) for stem_idx in range(k)]
    x_2c = torch.stack(x_2c, dim=0).to(device)

    p, _, _, _, _ = model(video, x_2c.view(1, 2, int(math.sqrt(x_dim)), int(math.sqrt(x_dim))))

    return torch.log(p.squeeze())


def separate_video(gt_m,
                   video,
                   hps_stems,
                   hps_video,
                   video_model_name,
                   stem_names,
                   L=10,
                   T=100,
                   alpha=1,
                   delta=2*1e-05,
                   image_h=64,
                   image_w=64,
                   sigma_start=0.01,
                   sigma_end=1.0,
                   visualise=False,
                   k=k, constraint_term_weight=-1,
                   verbose=True,
                   video_weight=1,
                   device=torch.device('cuda'),
                   gradient_weight=1):

    x_dim = image_h * image_w

    z_dim = hps_stems['hidden']

    gt_m_xz = torch.cat([gt_m.view(-1), torch.zeros(z_dim).to(device)]).to(device)

    vaes = get_vaes_rochester(stem_names, device)

    sigmas = torch.logspace(start=torch.log10(torch.tensor(sigma_start)),
                            end=torch.log10(torch.tensor(sigma_end)),
                            steps=L, base=10).flip(dims=[0]).to(device)

    # sigmas.requires_grad_(True)

    xz = []

    for i in range(k):
        noise_image = torch.rand(image_h, image_w).to(device)
        mu, log_var = vaes[i].encode(noise_image.unsqueeze(dim=0).unsqueeze(dim=0))
        z = vaes[i].reparameterise(mu, log_var)
        # z = torch.randn((1, z_dim)).to(device)
        xz.append(noise_image.view(-1).detach().cpu())
        xz.append(z.view(-1).detach().cpu())

    # create a big flat vector
    xz = torch.cat(xz).to(device)
    if video is not None:
        video = video.to(device)
        video_model = VideoModel(hps_video['z_dim_2d'], hps_video['z_dim_3d'], device=device).to(device)
        video_model.load_state_dict(torch.load(f'checkpoints/{video_model_name}.pth', map_location=device))
        video_model.eval()
        stacked_x = extract_stacked_x(xz, x_dim, z_dim).to(device)
        s, _, _, _, _ = video_model(video, stacked_x)
        s = s.squeeze(dim=1)
        s = torch.log(s)
    else:
        s = torch.tensor([0]).to(device)

    xz = torch.cat([xz, s])
    xz.requires_grad_(True)
    assert len(xz) == k * (x_dim + z_dim) + 1

    # for i, x in enumerate(gt_xs):
    #     save_image(x, f'images/000_gt_stem_{i}_new.png')
#
    # save_image(m, 'images/000_m.png')

    xz_chain = []

    # eta_0 = delta * sigmas[0] ** 2 / sigmas[L - 1] ** 2
    # eta_i_div_sigma_i_pow_2 = (eta_0 / sigmas[0] ** 2)

    for i in range(L):
        eta_i = delta * sigmas[i] ** 2 / sigmas[L - 1] ** 2

        for t in range(T):
            epsilon_t = torch.randn(xz.shape, requires_grad=True).to(device)

            if xz.grad is not None:
                xz.grad.zero_()

            if video is not None:
                log_prob3 = log_p_s(video_model, video, xz, x_dim, z_dim, k, device)
                log_prob3 = log_prob3 * video_weight
            else:
                log_prob3 = torch.tensor(0)

            log_prob1 = log_p_z(xz, x_dim=x_dim, z_dim=z_dim, k=k)
            log_prob2 = log_p_x_given_z(vaes, xz, sigmas[i], x_dim=x_dim, z_dim=z_dim, k=k, device=device)

            # elbo = vae.log_prob(xs[source_idx].unsqueeze(dim=0)).to(device)
            # grad_log_p_x = #torch.autograd.grad(elbo, xs[source_idx], retain_graph=True, create_graph=True)[0]
            log_p_x_z_s = log_prob1 + log_prob2 + log_prob3

            try:
                grad_log_p_x_z_s = torch.autograd.grad(log_p_x_z_s, xz)[0].detach() * gradient_weight
            except RuntimeError:
                print('error computing gradient')

            # print(eta_i)
            u = xz + eta_i * grad_log_p_x_z_s + torch.sqrt(2 * eta_i) * epsilon_t
            constraint_term = (eta_i / sigmas[i] ** 2) * (gt_m_xz - g_xz(xz, x_dim=x_dim, z_dim=z_dim, device=device)).float() * alpha
            elongated_constraint_term = torch.cat([constraint_term for _ in range(k)] + [torch.tensor([0]).to(device)]) * constraint_term_weight
            xz = u - elongated_constraint_term  # minmax_rows(u - temp)

            if xz[-1] > 0:
                xz[-1] = 0

            # print(torch.exp(xz[-1]))

            del epsilon_t, u, constraint_term, elongated_constraint_term, log_p_x_z_s, grad_log_p_x_z_s

        xz_chain.append(xz.detach().cpu())

        # for vis_idx in range(k):
            # x = extract_x(xz_chain[-1], vis_idx, x_dim=x_dim, z_dim=z_dim).view(image_h, image_w)
            # save_image(x, f'images/0_recon_stem_{vis_idx + 1}_gen{i}.png')

        if verbose:
            print(f'Appended {i + 1}/{L}')

        del eta_i

    final_xz = xz_chain[-1]
    final_xs = []

    for i in range(k):
        x = extract_x(final_xz, i, x_dim=x_dim, z_dim=z_dim).view(image_h, image_w)
        final_xs.append(x)
        if visualise:
            save_image(x, f'images/000_recon_stem_{i + 1}_new.png')

    if visualise:
        m_recon = g(final_xs)
        save_image(m_recon, 'images/000_m_recon_new.png')

    final_samples = [vaes[stem_idx].decode(extract_z(xz, stem_idx, x_dim=x_dim, z_dim=z_dim).unsqueeze(dim=0)) for stem_idx in range(k)]
    final_xs = [extract_x(xz, stem_idx, x_dim=x_dim, z_dim=z_dim) for stem_idx in range(k)]
    final_zs = [extract_z(xz, stem_idx, x_dim=x_dim, z_dim=z_dim) for stem_idx in range(k)]
    # print(final_zs)

    # clear memory from gpu
    del xz_chain, xz, gt_m_xz, sigmas, gt_m, vaes

    return final_xs


