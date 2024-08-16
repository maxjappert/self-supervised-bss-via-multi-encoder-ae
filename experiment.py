import sys
from datetime import datetime
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
from functions import get_non_linear_separation, model_factory, get_linear_separation, get_model
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

def experiment(name_vae):
    k = 2
    image_h = 64
    image_w = 64

    # name_vae = sys.argv[1]
    hps_vae = json.load(open(f'hyperparameters/{name_vae}_stem1.json'))

    model_bss = get_model(f'{name_vae}_separator', image_h=64, image_w=64, k=2)
    model_bss_linear = get_model(f'{name_vae}_separator_linear', image_h=64, image_w=64, k=2)

    debug = False

    dataset_name = 'toy_dataset' if name_vae.__contains__('toy') else 'musdb_18_prior'

    test_datasets = [
        PriorDataset('test', image_h=image_h, image_w=image_w, name=dataset_name, num_stems=4, debug=debug, stem_type=i + 1)
        for i in range(4)]

    print(len(test_datasets[0]))

    # total_basis_sdr_mireval_1 = 0
    # total_basis_sdr_mireval_2 = 0

    total_sample_sdr_1 = 0
    total_sample_sdr_2 = 0

    num_samples = 450

    metrics = {'sdr': 0,
               'isr': 1,
               'sir': 2,
               'sar': 3}

    hps_opti_toy = {'T': 34,
                    'alpha': 0.0006401233529860387,
                    'sigma_start': 0.051671063087265365,
                    'sigma_end': 0.5725567181888609,
                    'delta': 2.2945647974306533e-06,
                    'recon_weight': -45.3023273702013,
                    'L': 51}

    hps_opti_musdb = {'T': 64,
                      'alpha': 0.9684708911322072,
                      'sigma_start': 0.09090623169744898,
                      'sigma_end': 0.9164961200531399,
                      'delta': 5.972386539628251e-07,
                      'recon_weight': -42.86084692317908,
                      'L': 29}

    hps_opti = hps_opti_toy if name_vae.__contains__('toy') else hps_opti_musdb

    T = hps_opti['T']
    alpha = hps_opti['alpha']
    sigma_start = hps_opti['sigma_start']
    sigma_end = hps_opti['sigma_end']
    delta = hps_opti['delta']
    recon_weight = hps_opti['recon_weight']
    L = hps_opti['L']

    results_basis = np.zeros((k, len(metrics.keys()), num_samples))
    results_basis_finetuned = np.zeros((k, len(metrics.keys()), num_samples))
    results_basis_opti = np.zeros((k, len(metrics.keys()), num_samples))
    results_prior_samples = np.zeros((k, len(metrics.keys()), num_samples))
    results_noise = np.zeros((k, len(metrics.keys()), num_samples))
    results_bss = np.zeros((k, len(metrics.keys()), num_samples))
    results_bss_linear = np.zeros((k, len(metrics.keys()), num_samples))
    results_nmf = np.zeros((k, len(metrics.keys()), num_samples))

    for i in range(num_samples):
        stem_indices = [random.randint(0, 3), random.randint(0, 3)] # [0, 3]
        vaes = get_vaes(name_vae, stem_indices)
        now = datetime.now()
        timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")
        print(f'[{timestamp_str}]  Processing {i+1}/{num_samples}')
        # to avoid, when the same stem is selected, the same sample
        gt_data = [test_datasets[stem_index][i + j] for j, stem_index in enumerate(stem_indices)]
        gt_xs = [data['spectrogram'] for data in gt_data]

        gt_m = torch.sum(torch.cat(gt_xs), dim=0).to(device)

        # separate using basis
        separated_basis = separate(gt_m, hps_vae, name=name_vae, stem_indices=stem_indices, finetuned=False, alpha=1, visualise=False, verbose=False,
                                   constraint_term_weight=-15)
        separated_basis = [x_i.detach().cpu() for x_i in separated_basis]

        separated_basis_finetuned = separate(gt_m, hps_vae, name=name_vae, stem_indices=stem_indices, finetuned=True, alpha=1, visualise=False, verbose=False, sigma_end=0.5 if name_vae.__contains__('toy') else 1.0,
                                   constraint_term_weight=-18)
        separated_basis_finetuned = [x_i.detach().cpu() for x_i in separated_basis_finetuned]

        separated_basis_opti = separate(gt_m,
                                        hps_vae,
                                        name=name_vae,
                                        stem_indices=stem_indices,
                                        finetuned=False,
                                        alpha=alpha,
                                        visualise=False,
                                        verbose=False,
                                        sigma_start=sigma_start,
                                        sigma_end=sigma_end,
                                        constraint_term_weight=recon_weight,
                                        T=T,
                                        L=L,
                                        delta=delta)
        separated_basis_opti = [x_i.detach().cpu() for x_i in separated_basis_opti]

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
        sdr_basis_finetuned, isr_basis_finetuned, sir_basis_finetuned, sar_basis_finetuned, _ = mir_eval.separation.bss_eval_images(gt_xs, separated_basis_finetuned)
        sdr_basis_opti, isr_basis_opti, sir_basis_opti, sar_basis_opti, _ = mir_eval.separation.bss_eval_images(gt_xs, separated_basis_opti)
        sdr_sample, isr_sample, sir_sample, sar_sample, _ = mir_eval.separation.bss_eval_images(gt_xs, samples)
        sdr_noise, isr_noise, sir_noise, sar_noise, _ = mir_eval.separation.bss_eval_images(gt_xs, noise_images)
        sdr_bss, isr_bss, sir_bss, sar_bss, _ = mir_eval.separation.bss_eval_images(gt_xs, separated_bss)
        sdr_bss_linear, isr_bss_linear, sir_bss_linear, sar_bss_linear, _ = mir_eval.separation.bss_eval_images(gt_xs, separated_bss_linear)
        sdr_nmf, isr_nmf, sir_nmf, sar_nmf, _ = mir_eval.separation.bss_eval_images(gt_xs, nmf_recons)

        # put metrics into relevant location for evaluation later
        results_basis[:, :, i] = np.array([sdr_basis, isr_basis, sir_basis, sar_basis]).T
        results_basis_finetuned[:, :, i] = np.array([sdr_basis_finetuned, isr_basis_finetuned, sir_basis_finetuned, sar_basis_finetuned]).T
        results_basis_opti[:, :, i] = np.array([sdr_basis_opti, isr_basis_opti, sir_basis_opti, sar_basis_opti]).T
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
    print_output(results_basis_finetuned, k, 0, 'BASIS Finetuned')
    print_output(results_basis_opti, k, 0, 'BASIS Opti')
    print_output(results_prior_samples, k, 0, 'Prior samples')
    print_output(results_noise, k, 0, 'Noise')
    print_output(results_bss, k, 0, 'BSS')
    print_output(results_bss_linear, k, 0, 'Linear BSS')
    print_output(results_nmf, k, 0, 'NMF')

    np.save(f'results/results_basis_{name_vae}.npy', results_basis)
    np.save(f'results/results_basis_finetuned_{name_vae}.npy', results_basis_finetuned)
    np.save(f'results/results_basis_opti_{name_vae}.npy', results_basis_opti)
    np.save(f'results/results_prior_samples_{name_vae}.npy', results_prior_samples)
    np.save(f'results/results_noise_{name_vae}.npy', results_noise)
    np.save(f'results/results_bss_{name_vae}.npy', results_bss)
    np.save(f'results/results_bss_linear_{name_vae}.npy', results_bss_linear)
    np.save(f'results/results_nmf_{name_vae}.npy', results_nmf)

       # print(f'Weight: {constraint_term_weight}')
        # print(f'{round(np.mean(basis_sdr_1), 3)} +- {round(np.std(basis_sdr_1), 3)}')
        # print(f'{round(np.mean(basis_sdr_2), 3)} +- {round(np.std(basis_sdr_2), 3)}')
        # print()
