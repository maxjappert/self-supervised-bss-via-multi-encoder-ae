from datetime import datetime
import json
import random
import sys

import mir_eval
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import optim
from torch.nn import MSELoss
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os

from functions_prior import VAE, PriorDataset, finetune_sigma, train_vae

# Define the seed
seed = 42

# Set the seed for Python's built-in random module
random.seed(seed)

# Set the seed for NumPy
np.random.seed(seed)

# Set the seed for PyTorch
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

# For CuDNN backend (optional, but recommended for reproducibility)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def extract_x(xz, stem_idx, x_dim, z_dim):
    start_idx = stem_idx * (x_dim + z_dim)
    end_idx = start_idx + x_dim
    return xz[start_idx:end_idx]


def extract_z(xz, stem_idx, x_dim, z_dim):
    start_idx = (stem_idx + 1) * x_dim + stem_idx * z_dim
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


def g(stems, alpha=1):
    return torch.sum(torch.stack(stems, dim=0) * alpha if type(stems) is list else stems * alpha, dim=0)


def g_xz(xz, x_dim, z_dim, device):
    total_sum = 0

    for stem_idx in range(k):
        total_sum += torch.cat([extract_x(xz, stem_idx, x_dim, z_dim).to(device), torch.zeros(z_dim).to(device)])

    return total_sum


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# name = 'musdb_small_newelbo'
k = 2

debug = False

def finetune_sigma_models(vae, dataloader_train, dataloader_val, sigmas):
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
    name = sys.argv[3]
    hps = json.load(open(f'hyperparameters/{name}_stem1.json'))
    image_h = hps['image_h']
    image_w = hps['image_w']

    vae = VAE(latent_dim=hps['hidden'],
              image_h=image_h,
              image_w=image_w,
              use_blocks=hps['use_blocks'],
              channels=hps['channels'],
              kernel_sizes=hps['kernel_sizes'],
              strides=hps['strides']).to(device)

    stem_type = int(sys.argv[2])

    dataset_name = 'toy_dataset' if name == 'toy' else 'musdb_18_prior'

    assert stem_type in range(1, 5)

    train_dataset = PriorDataset('train', image_h=image_h, image_w=image_w, name=dataset_name, num_stems=4,
                                 debug=debug,
                                 stem_type=stem_type)

    val_dataset = PriorDataset('val', image_h=image_h, image_w=image_w, name=dataset_name, num_stems=4, debug=debug,
                               stem_type=stem_type)

    dataloader_train = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4)
    dataloader_val = DataLoader(val_dataset, batch_size=256, shuffle=True, num_workers=4)

    vae_name = f'{name}_stem{stem_type}'
    vae.load_state_dict(torch.load(f'checkpoints/{vae_name}.pth', map_location=device))

    sigma_start = 0.01
    sigma_end = 1.0
    L = 10
    sigmas = torch.logspace(start=torch.log10(torch.tensor(sigma_start)),
                            end=torch.log10(torch.tensor(sigma_end)),
                            steps=L, base=10).flip(dims=[0])

    finetune_sigma_models(vae, dataloader_train, dataloader_val, sigmas)

    sys.exit(0)


def get_vaes(name, stem_indices, device, sigma=None):
    hps = json.load(open(os.path.join('hyperparameters', f'{name}_stem1.json')))
    vaes = []

    for stem_index in stem_indices:
        vae = VAE(latent_dim=hps['hidden'],
                          image_h=64,
                          image_w=64,
                          use_blocks=hps['use_blocks'],
                          channels=hps['channels'],
                          kernel_sizes=hps['kernel_sizes'],
                          strides=hps['strides']).to(device)

        vae_name = f'{name}_stem{stem_index + 1}' if sigma is None else f'sigma_{name}_stem{stem_index + 1}_{sigma}'

        # vae_name = f'{name}_stem{stem_type + 1}'
        vae.load_state_dict(torch.load(f'checkpoints/{vae_name}.pth', map_location=device))

        vaes.append(vae)

    return vaes

def get_vaes_rochester(names, device):
    vaes = []

    names = [f'{name}_{name}' for name in names]

    for name in names:
        hps = json.load(open(os.path.join('hyperparameters', f'{name}.json')))

        vae = VAE(latent_dim=hps['hidden'],
                          image_h=64,
                          image_w=64,
                          use_blocks=hps['use_blocks'],
                          channels=hps['channels'],
                          kernel_sizes=hps['kernel_sizes'],
                          strides=hps['strides']).to(device)

        # vae_name = f'{name}_stem{stem_index + 1}' if sigma is None else f'sigma_{name}_stem{stem_index + 1}_{sigma}'

        # vae_name = f'{name}_stem{stem_type + 1}'
        vae.load_state_dict(torch.load(f'checkpoints/{name}.pth', map_location=device))

        vaes.append(vae)

    return vaes


def log_p_z(xz, x_dim, z_dim):
    total_log_prob = 0
    mvsn = torch.distributions.normal.Normal(0, 1)

    for stem_idx in range(k):
        z = extract_z(xz, stem_idx, x_dim=x_dim, z_dim=z_dim)

        assert len(z) == z_dim

        total_log_prob += torch.sum(mvsn.log_prob(z))

    return total_log_prob


def log_p_x_given_z(vaes, xz, sigma, x_dim, z_dim):
    total_log_prob = 0

    for stem_idx in range(k):
        x = extract_x(xz, stem_idx, x_dim=x_dim, z_dim=z_dim)
        z = extract_z(xz, stem_idx, x_dim=x_dim, z_dim=z_dim)

        x_recon = vaes[stem_idx].decode(z.unsqueeze(dim=0))
        mvn = torch.distributions.normal.Normal(x_recon.view(-1), sigma)

        total_log_prob += torch.sum(mvn.log_prob(x))

        del x_recon, mvn, x, z

    return total_log_prob

def separate(gt_m,
             hps,
             L=10,
             T=100,
             alpha=1,
             delta=2*1e-05,
             image_h=64,
             image_w=64,
             sigma_start=0.01,
             sigma_end=1.0,
             stem_indices=[0, 3],
             finetuned=True,
             name=None,
             visualise=False,
             k=k, constraint_term_weight=-1,
             verbose=True,
             gradient_weight=1,
             device=device):

    x_dim = image_h * image_w

    z_dim = hps['hidden']

    # stem_indices = [0, 3]
    # gt_xs = [val_datasets[stem_index][random.randint(0, 100)]['spectrogram'] for stem_index in stem_indices]

    gt_m_xz = torch.cat([gt_m.view(-1), torch.zeros(z_dim).to(device)]).to(device)

    vaes_perfect = get_vaes(name, stem_indices, device)

    if finetuned:
        vaes_noisy = get_vaes(name, stem_indices, device, sigma=sigma_end)
    else:
        vaes_noisy = vaes_perfect

    sigmas = torch.logspace(start=torch.log10(torch.tensor(sigma_start)),
                            end=torch.log10(torch.tensor(sigma_end)),
                            steps=L, base=10).flip(dims=[0]).to(device)

    # sigmas.requires_grad_(True)

    xz = []

    for i in range(k):
        noise_image = torch.rand(image_h, image_w).to(device)
        mu, log_var = vaes_noisy[i].encode(noise_image.unsqueeze(dim=0).unsqueeze(dim=0))
        z = vaes_noisy[i].reparameterise(mu, log_var)
        # z = torch.randn((1, z_dim)).to(device)
        xz.append(noise_image.view(-1))
        xz.append(z.view(-1))

    # create a big flat vector
    xz = torch.cat(xz).to(device)
    xz.requires_grad_(True)
    assert len(xz) == k * (x_dim + z_dim), f'{len(xz)}, {k}, {x_dim}, {z_dim}'

    # for i, x in enumerate(gt_xs):
    #     save_image(x, f'images/000_gt_stem_{i}_new.png')
#
    # save_image(m, 'images/000_m.png')

    xz_chain = []

    # eta_0 = delta * sigmas[0] ** 2 / sigmas[L - 1] ** 2
    # eta_i_div_sigma_i_pow_2 = (eta_0 / sigmas[0] ** 2)

    for i in range(L):
        eta_i = delta * sigmas[i] ** 2 / sigmas[L - 1] ** 2
        sigma_vaes = []

        if finetuned:
            for stem_index in stem_indices:
                    sigma_model_name = f'sigma_{name}_stem{stem_index + 1}_{np.round(sigmas[i].detach().item(), 3)}'
                    vae = VAE(z_dim, image_h=image_h, image_w=image_w, use_blocks=hps['use_blocks'],
                              channels=hps['channels'], kernel_sizes=hps['kernel_sizes'], strides=hps['strides']).to(
                        device)
                    vae.load_state_dict(torch.load(f'checkpoints/{sigma_model_name}.pth', map_location=device))
                    sigma_vaes.append(vae)
        else:
            sigma_vaes = vaes_perfect

        for t in range(T):

            epsilon_t = torch.randn(xz.shape).to(device)

            if xz.grad is not None:
                xz.grad.zero_()

            # elbo = vae.log_prob(xs[source_idx].unsqueeze(dim=0)).to(device)
            # grad_log_p_x = #torch.autograd.grad(elbo, xs[source_idx], retain_graph=True, create_graph=True)[0]
            log_p_x_z = log_p_z(xz, x_dim=x_dim, z_dim=z_dim) + log_p_x_given_z(sigma_vaes, xz, sigmas[i], x_dim=x_dim, z_dim=z_dim)

            grad_log_p_x_z = torch.autograd.grad(log_p_x_z, xz)[0].detach() * gradient_weight

            # print(eta_i)
            u = xz + eta_i * grad_log_p_x_z + torch.sqrt(2 * eta_i) * epsilon_t
            constraint_term = (eta_i / sigmas[i] ** 2) * (gt_m_xz - g_xz(xz, x_dim=x_dim, z_dim=z_dim, device=device)).float() * alpha
            elongated_constraint_term = torch.cat([constraint_term for _ in range(k)]) * constraint_term_weight
            xz = u - elongated_constraint_term  # minmax_rows(u - temp)

            del epsilon_t, u, constraint_term, elongated_constraint_term, log_p_x_z, grad_log_p_x_z


        xz_chain.append(xz.cpu())

        # for vis_idx in range(k):
            # x = extract_x(xz_chain[-1], vis_idx, x_dim=x_dim, z_dim=z_dim).view(image_h, image_w)
            # save_image(x, f'images/0_recon_stem_{vis_idx + 1}_gen{i}.png')

        if verbose:
            print(f'Appended {i + 1}/{L}')

        for model in sigma_vaes:
            del model

        del sigma_vaes, eta_i
        torch.cuda.empty_cache()

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

    final_samples = [vaes_perfect[stem_idx].decode(extract_z(xz, stem_idx, x_dim=x_dim, z_dim=z_dim).unsqueeze(dim=0)) for stem_idx in range(k)]
    final_xs = [extract_x(xz, stem_idx, x_dim=x_dim, z_dim=z_dim) for stem_idx in range(k)]
    final_zs = [extract_z(xz, stem_idx, x_dim=x_dim, z_dim=z_dim) for stem_idx in range(k)]
    # print(final_zs)

    # clear memory from gpu
    del xz_chain, xz, gt_m_xz, sigmas, gt_m, vaes_noisy, vaes_perfect

    return final_xs


def evaluate_basis_ability(recon_weight, gradient_weight, image_h=64, image_w=64, num_samples=10, name_vae='toy', finetuned=False):
    hps_vae = json.load(open(f'hyperparameters/{name_vae}_stem1.json'))

    dataset_name = 'toy_dataset' if name_vae.__contains__('toy') else 'musdb_18_prior'

    val_datasets = [
        PriorDataset('val', image_h=image_h, image_w=image_w, name=dataset_name, num_stems=4, debug=debug,
                     stem_type=i + 1)
        for i in range(4)]

    total_sdr = 0

    for i in range(num_samples):
        stem_indices = [random.randint(0, 3), random.randint(0, 3)] # [0, 3]
        now = datetime.now()
        timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")
        # print(f'[{timestamp_str}]  Processing {i+1}/{num_samples}')
        # to avoid, when the same stem is selected, the same sample
        gt_data = [val_datasets[stem_index][i + j] for j, stem_index in enumerate(stem_indices)]
        gt_xs = [data['spectrogram'] for data in gt_data]

        gt_m = torch.sum(torch.cat(gt_xs), dim=0).to(device)

        # separate using basis
        separated_basis = separate(gt_m,
                                   hps_vae,
                                   name=name_vae,
                                   stem_indices=stem_indices,
                                   finetuned=finetuned,
                                   visualise=False,
                                   verbose=False,
                                   constraint_term_weight=recon_weight,
                                   gradient_weight=gradient_weight)
        separated_basis = [x_i.detach().cpu() for x_i in separated_basis]

        gt_m = gt_m.cpu()

        gt_xs = np.array([x.squeeze().view(-1) for x in gt_xs])

        sdr, isr, sir, sar, _ = mir_eval.separation.bss_eval_images(gt_xs, separated_basis)

        total_sdr += sdr.mean()

        torch.cuda.empty_cache()

    return total_sdr / num_samples

