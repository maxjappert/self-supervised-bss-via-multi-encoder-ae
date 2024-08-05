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
from functions_prior import PriorDataset
from separate_new import separate, get_vaes

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

np.set_printoptions(precision=3, suppress=True)

name = 'toy'
hps = json.load(open(f'hyperparameters/{name}_stem1.json'))
image_h = hps['image_h']
image_w = hps['image_w']

k = 2

debug = False


test_datasets = [
    PriorDataset('test', image_h=image_h, image_w=image_w, name='toy_dataset', num_stems=4, debug=debug, stem_type=i + 1)
    for i in range(4)]

# total_basis_sdr_mireval_1 = 0
# total_basis_sdr_mireval_2 = 0

total_sample_sdr_1 = 0
total_sample_sdr_2 = 0
stem_indices = [0, 3]

num_samples = 10

metrics = {'sdr': 0,
           'isr': 1,
           'sir': 2,
           'sar': 3}

basis = np.zeros((k, len(metrics.keys()), num_samples))
prior_samples = np.zeros((k, len(metrics.keys()), num_samples))

for i in range(num_samples):
    gt_data = [test_datasets[stem_index][i + 15] for stem_index in stem_indices]
    gt_xs = [data['spectrogram'] for data in gt_data]

    gt_m = torch.sum(torch.cat(gt_xs), dim=0)

    # separate using basis
    separated_basis = separate(gt_m, hps['hidden'], name=name, finetuned=False, alpha=1, visualise=False, verbose=False,
                               constraint_term_weight=-4)
    separated_basis = [x_i.detach().cpu() for x_i in separated_basis]

    # draw sample from prior
    vaes = get_vaes(name, stem_indices)
    samples = [vae.decode(torch.randn(1, hps['hidden']).to(device)).squeeze().detach().cpu() for vae in vaes]
    samples = [x_i.detach().cpu() for x_i in samples]

    # create noisy image
    noise_images = [torch.rand_like(samples[i]) for i in range(k)]

    # evaluate using nmf
    nmf_recons = nmf_approx_two_sources(gt_m)
    nmf_recons = [x_i.view(-1) for x_i in nmf_recons]

    gt_xs = np.array([x.squeeze().view(-1) for x in gt_xs])

    # compute metrics
    sdr_basis, isr_basis, sir_basis, sar_basis, _ = mir_eval.separation.bss_eval_images(gt_xs, separated_basis)
    sdr_sample, isr_sample, sir_sample, sar_sample, _ = mir_eval.separation.bss_eval_images(gt_xs, samples)
    sdr_noise, isr_noise, sir_noise, sar_noise, _ = mir_eval.separation.bss_eval_images(gt_xs, noise_images)

    # put metrics into relevant location for evaluation later
    basis[:, :, i] = np.array([sdr_basis, isr_basis, sir_basis, sar_basis]).T
    prior_samples[:, :, i] = np.array([sdr_sample, isr_sample, sir_sample, sar_sample]).T

    print(f'Weight: {constraint_term_weight}')
    print(f'{round(np.mean(basis_sdr_1), 3)} +- {round(np.std(basis_sdr_1), 3)}')
    print(f'{round(np.mean(basis_sdr_2), 3)} +- {round(np.std(basis_sdr_2), 3)}')
    print()
