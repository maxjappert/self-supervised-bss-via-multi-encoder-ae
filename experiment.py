import json
import random

import mir_eval
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from evaluate_nmf import nmf_approx_two_sources
from evaluation_metric_functions import compute_spectral_sdr, compute_spectral_metrics
from functions import get_non_linear_separation, model_factory, get_linear_separation
from functions_prior import PriorDataset
from separate_new import separate, get_vaes

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

np.set_printoptions(precision=3, suppress=True)

k = 2
image_h = 64
image_w = 64

name_vae = 'toy'
hps_vae = json.load(open(f'hyperparameters/{name_vae}_stem1.json'))
image_h = hps_vae['image_h']
image_w = hps_vae['image_w']

name_bss = 'toy_separator'
hps_bss = json.load(open(f'hyperparameters/{name_bss}.json'))
model_bss = model_factory(linear=hps_bss['linear'],
                          channels=hps_bss['channels'],
                          hidden=hps_bss['hidden'],
                          num_encoders=k,
                          image_height=image_h,
                          image_width=image_w,
                          norm_type=hps_bss['norm_type'],
                          use_weight_norm=hps_bss['use_weight_norm'],
                          kernel_size=hps_bss['kernel_size']).to(device)

model_bss.load_state_dict(torch.load(f'checkpoints/{name_bss}.pth', map_location=device))

name_bss_linear = 'toy_separator_linear'
hps_bss_linear = json.load(open(f'hyperparameters/{name_bss_linear}.json'))
model_bss_linear = model_factory(linear=hps_bss_linear['linear'],
                                 channels=hps_bss_linear['channels'],
                                 hidden=hps_bss_linear['hidden'],
                                 num_encoders=k,
                                 image_height=image_h,
                                 image_width=image_w,
                                 norm_type=hps_bss_linear['norm_type'],
                                 use_weight_norm=hps_bss_linear['use_weight_norm'],
                                 kernel_size=hps_bss_linear['kernel_size']).to(device)

model_bss_linear.load_state_dict(torch.load(f'checkpoints/{name_bss_linear}.pth', map_location=device))

debug = False

test_datasets = [
    PriorDataset('test', image_h=image_h, image_w=image_w, name='toy_dataset', num_stems=4, debug=debug, stem_type=i + 1)
    for i in range(4)]

# total_basis_sdr_mireval_1 = 0
# total_basis_sdr_mireval_2 = 0

total_sample_sdr_1 = 0
total_sample_sdr_2 = 0
stem_indices = [0, 3]

num_samples = 900

metrics = {'sdr': 0,
           'isr': 1,
           'sir': 2,
           'sar': 3}

results_basis = np.zeros((k, len(metrics.keys()), num_samples))
results_prior_samples = np.zeros((k, len(metrics.keys()), num_samples))
results_noise = np.zeros((k, len(metrics.keys()), num_samples))
results_bss = np.zeros((k, len(metrics.keys()), num_samples))
results_bss_linear = np.zeros((k, len(metrics.keys()), num_samples))
results_nmf = np.zeros((k, len(metrics.keys()), num_samples))

vaes = get_vaes(name_vae, stem_indices)

for i in range(num_samples):
    print(f'Processing {i+1}/{num_samples}')
    gt_data = [test_datasets[stem_index][i + 15] for stem_index in stem_indices]
    gt_xs = [data['spectrogram'] for data in gt_data]

    gt_m = torch.sum(torch.cat(gt_xs), dim=0)

    # separate using basis
    separated_basis = separate(gt_m, hps_vae['hidden'], name=name_vae, finetuned=False, alpha=1, visualise=False, verbose=False,
                               constraint_term_weight=-15)
    separated_basis = [x_i.detach().cpu() for x_i in separated_basis]

    # draw sample from prior
    samples = [vae.decode(torch.randn(1, hps_vae['hidden']).to(device)).squeeze().detach().cpu() for vae in vaes]
    samples = np.array([x_i.detach().cpu().view(-1) for x_i in samples])

    # create noisy image
    noise_images = np.array([torch.rand_like(torch.tensor(samples[i])) for i in range(k)])

    # evaluate using nmf
    nmf_recons = nmf_approx_two_sources(gt_m)
    nmf_recons = np.array([torch.tensor(x_i).view(-1) for x_i in nmf_recons])

    # get vae_bss separation
    separated_bss = get_non_linear_separation(model_bss, gt_m.unsqueeze(0).unsqueeze(0))
    separated_bss = np.array([torch.tensor(x_i).view(-1) for x_i in separated_bss])

    separated_bss_linear = get_linear_separation(model_bss_linear, gt_m.unsqueeze(0).unsqueeze(0))
    separated_bss_linear = np.array([torch.tensor(x_i).view(-1) for x_i in separated_bss_linear])

    gt_xs = np.array([x.squeeze().view(-1) for x in gt_xs])

    # compute metrics
    sdr_basis, isr_basis, sir_basis, sar_basis, _ = mir_eval.separation.bss_eval_images(gt_xs, separated_basis)
    sdr_sample, isr_sample, sir_sample, sar_sample, _ = mir_eval.separation.bss_eval_images(gt_xs, samples)
    sdr_noise, isr_noise, sir_noise, sar_noise, _ = mir_eval.separation.bss_eval_images(gt_xs, noise_images)
    sdr_bss, isr_bss, sir_bss, sar_bss, _ = mir_eval.separation.bss_eval_images(gt_xs, separated_bss)
    sdr_bss_linear, isr_bss_linear, sir_bss_linear, sar_bss_linear, _ = mir_eval.separation.bss_eval_images(gt_xs, separated_bss_linear)
    sdr_nmf, isr_nmf, sir_nmf, sar_nmf, _ = mir_eval.separation.bss_eval_images(gt_xs, nmf_recons)

    # put metrics into relevant location for evaluation later
    results_basis[:, :, i] = np.array([sdr_basis, isr_basis, sir_basis, sar_basis]).T
    results_prior_samples[:, :, i] = np.array([sdr_sample, isr_sample, sir_sample, sar_sample]).T
    results_noise[:, :, i] = np.array([sdr_noise, isr_noise, sir_noise, sar_noise]).T
    results_bss[:, :, i] = np.array([sdr_bss, isr_bss, sir_bss, sar_bss]).T
    results_bss_linear[:, :, i] = np.array([sdr_bss_linear, isr_bss_linear, sir_bss_linear, sar_bss_linear]).T
    results_nmf[:, :, i] = np.array([sdr_nmf, isr_nmf, sir_nmf, sar_nmf]).T


def print_output(results, k, metric_idx, name):
    for stem_idx in range(k):
        print(f'{name} {stem_idx+1}: {round(np.mean(results[stem_idx, metric_idx, :]), 3)} +- {round(np.std(results[stem_idx, metric_idx, :]), 3)}')

    print()


print_output(results_basis, k, 0, 'BASIS')
print_output(results_prior_samples, k, 0, 'Prior samples')
print_output(results_noise, k, 0, 'Noise')
print_output(results_bss, k, 0, 'BSS')
print_output(results_bss_linear, k, 0, 'Linear BSS')
print_output(results_nmf, k, 0, 'NMF')

np.save('results/results_basis.npy', results_basis)
np.save('results/results_prior_samples.npy', results_prior_samples)
np.save('results/results_noise.npy', results_noise)
np.save('results/results_bss.npy', results_bss)
np.save('results/results_bss_linear.npy', results_bss_linear)
np.save('results/results_nmf.npy', results_nmf)

   # print(f'Weight: {constraint_term_weight}')
    # print(f'{round(np.mean(basis_sdr_1), 3)} +- {round(np.std(basis_sdr_1), 3)}')
    # print(f'{round(np.mean(basis_sdr_2), 3)} +- {round(np.std(basis_sdr_2), 3)}')
    # print()
