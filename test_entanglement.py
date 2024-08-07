import json

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from functions_prior import PriorDataset, VAE
import torch.distributions as dist

debug = True
device = torch.device('cpu')

names = ['musdb_small_newelbo', 'toy_small_newelbo']

def evaluate_parameters():
    for name in names:
        print(name)

        if name.__contains__('toy'):
            dataset_name = 'toy_dataset'
        else:
            dataset_name = 'musdb_18_prior'

        hps = json.load(open(f'hyperparameters/{name}.json'))

        image_h = hps['image_h']
        image_w = hps['image_w']

        dataset_val = PriorDataset('val', debug=debug, name=dataset_name, image_w=image_w, image_h=image_h)
        dataloader_val = DataLoader(dataset_val, num_workers=12, batch_size=16, shuffle=True)

        mus = [[], [], [], []]
        vars = [[], [], [], []]

        vae = VAE(use_blocks=False, latent_dim=hps['hidden'], image_h=image_h, image_w=image_w, kernel_sizes=hps['kernel_sizes'], channels=hps['channels'], strides=hps['strides']).to(
            device)

        vae.load_state_dict(torch.load(f'checkpoints/{name}.pth', map_location=device))

        with torch.no_grad():
            for batch in dataloader_val:
                vae.eval()
                spectrograms = batch['spectrogram'].to(device)
                labels = batch['label'].to(device)

                recon, mu, logvar = vae(spectrograms.float())

                for i, label in enumerate(labels):
                    mus[label-1].append(mu[i])
                    vars[label-1].append(torch.exp(logvar[i]))

        for i in range(4):
            mus_tensor = torch.stack(mus[i], dim=0)
            vars_tensor = torch.stack(vars[i], dim=0)
            print(f'Class {i+1}: Mu = {torch.mean(mus_tensor, dim=0)+- torch.std(mus_tensor, dim=0)}, Logvar = {torch.mean(vars_tensor, dim=0) +- torch.std(vars_tensor, dim=0)}')

            torch.save({'mean': torch.mean(mus_tensor, dim=0), 'var': torch.mean(vars_tensor, dim=0)}, f'checkpoints/class{i+1}_params.pth')


# Function to create and display noise images
def create_noise_images(num_images, height, width):
    # Create a list to hold the noise images
    noise_images = []

    for _ in range(num_images):
        noise_image = torch.rand(height, width).unsqueeze(dim=0)
        noise_images.append(noise_image)

    return noise_images


if __name__ == "__main__":
    name = 'musdb_small_newelbo'

    hps = json.load(open(f'hyperparameters/{name}.json'))

    image_h = hps['image_h']
    image_w = hps['image_w']

    dataset_val = PriorDataset('val', debug=debug, name='toy_dataset', image_w=image_w, image_h=image_h)
    dataloader_val = DataLoader(dataset_val, num_workers=12, batch_size=1, shuffle=True)

    vae = VAE(use_blocks=False, latent_dim=hps['hidden'], image_h=image_h, image_w=image_w, kernel_sizes=hps['kernel_sizes'], channels=hps['channels'], strides=hps['strides']).to(
            device)

    vae.load_state_dict(torch.load(f'checkpoints/{name}.pth', map_location=device))

    kls = []

    noise_images = create_noise_images(16,  image_h, image_w)

    kl_divs_random = []
    log_probs_random = []

    vae.eval()
    for noise_image in noise_images:
        recon, mu, logvar = vae(noise_image.unsqueeze(dim=0))

        output_dist = dist.Normal(mu, torch.exp(0.5 * logvar))
        standard_normal = dist.Normal(0, 1)

        kl_div = dist.kl_divergence(output_dist, standard_normal)

        kl_divs_random.append(kl_div.squeeze())

        log_prob = vae.log_prob(noise_image.unsqueeze(dim=0))
        log_probs_random.append(log_prob.detach())

    kl_divs_tensor = torch.stack(kl_divs_random, dim=0)
    print(f'Random: {np.mean(log_probs_random)}')
    print(f'Random: {torch.mean(kl_divs_tensor, dim=0)}')

    kl_divs_vae = []
    log_probs_vae = []

    for batch in dataloader_val:
        spectrograms = batch['spectrogram'].to(device)
        labels = batch['label'].to(device)

        recon, mu, logvar = vae(spectrograms.float())

        # Create normal distributions
        output_dist = dist.Normal(mu, torch.exp(0.5 * logvar))
        standard_normal = dist.Normal(0, 1)

        kl_div = dist.kl_divergence(output_dist, standard_normal)

        kl_divs_vae.extend(list(torch.unbind(kl_div, dim=0)))

        log_prob = vae.log_prob(spectrograms.float())
        log_probs_random.append(log_prob.detach())

    kl_divs_vae_tensor = torch.stack(kl_divs_vae, dim=0)

    print(f'VAE: {torch.mean(kl_divs_vae_tensor, dim=0)}')
    print(f'VAE: {np.mean(log_probs_random)}')
