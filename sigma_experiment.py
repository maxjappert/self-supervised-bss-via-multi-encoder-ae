import json
import random
import time

import mir_eval
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision.utils import save_image

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

train_datasets = [PriorDataset('train', image_h=image_h, image_w=image_w, name='toy_dataset', num_stems=4, debug=debug,
                               stem_type=i + 1) for i in range(4)]
val_datasets = [
    PriorDataset('val', image_h=image_h, image_w=image_w, name='toy_dataset', num_stems=4, debug=debug, stem_type=i + 1)
    for i in range(4)]

dataloaders_train = [DataLoader(train_datasets[i], batch_size=256, shuffle=True, num_workers=4) for i in range(4)]
dataloaders_val = [DataLoader(val_datasets[i], batch_size=256, shuffle=True, num_workers=4) for i in range(4)]

# total_basis_sdr_mireval_1 = 0
# total_basis_sdr_mireval_2 = 0

total_sample_sdr_1 = 0
total_sample_sdr_2 = 0

num_samples = 50

metrics = {'sdr': 0,
           'isr': 1,
           'sir': 2,
           'sar': 3}

sigmas_L = [0.06, 0.07, 0.08, 0.09, 0.1]
basis = np.zeros((len(sigmas_L), k, len(metrics.keys()), num_samples))

for weight_index, sigma_L in enumerate(sigmas_L):
    prior_samples = np.zeros((k, len(metrics.keys()), num_samples))

    for i in range(num_samples):
        stem_indices = [random.randint(0, 3), random.randint(0, 3)]
        # print(i)
        gt_data = [val_datasets[stem_index][i+15] for stem_index in stem_indices]
        gt_xs = [data['spectrogram'] for data in gt_data]

        gt_m = torch.sum(torch.cat(gt_xs), dim=0)

        separated = separate(gt_m, hps['hidden'], name=name, finetuned=False, alpha=1, visualise=False, verbose=False, sigma_start=sigma_L, constraint_term_weight=-15)

        separated_1 = separated[0].detach().cpu().reshape((64, 64))
        separated_2 = separated[1].detach().cpu().reshape((64, 64))

        time1 = time.time()
        sdr, isr, sir, sar, perm = mir_eval.separation.bss_eval_images(np.array([x.squeeze().view(-1) for x in gt_xs]), np.array([separated_1.view(-1), separated_2.view(-1)]), compute_permutation=True)
        time2 = time.time()
        # print(f'One metric took {time2 - time1} seconds')

        basis[weight_index, :, :, i] = np.array([sdr, isr, sir, sar]).T

    print(f'sigma_L: {sigma_L}')
    print(f'{round(np.mean(basis[weight_index, 0, metrics["sdr"], :]), 3)} +- {round(np.std(basis[weight_index, 0, metrics["sdr"], :]), 3)}')
    print(f'{round(np.mean(basis[weight_index, 1, metrics["sdr"], :]), 3)} +- {round(np.std(basis[weight_index, 1, metrics["sdr"], :]), 3)}')
    print()

np.save('results/sigma_Ls_evaluated.npy', sigmas_L)
np.save('results/basis_sigma_toy_experiment_results.npy', basis)


# save_image(gt_xs[0], f'images/0_gt0.png')
# save_image(gt_xs[1], f'images/0_gt1.png')
# save_image(separated_1, f'images/0_recon0.png')
# save_image(separated_2, f'images/0_recon1.png')
#
# print(compute_spectral_sdr(gt_xs[0].squeeze(), separated_1))
# print(compute_spectral_sdr(gt_xs[1].squeeze(), separated_2))
#
# print(compute_spectral_sdr(gt_xs[0].squeeze(), separated_2))
# print(compute_spectral_sdr(gt_xs[1].squeeze(), separated_1))
#
# print(compute_spectral_sdr(gt_xs[0].squeeze(), gt_m))
# print(compute_spectral_sdr(gt_xs[1].squeeze(), gt_m))
#
# print(compute_spectral_sdr(gt_xs[0].squeeze(), torch.rand_like(gt_xs[0].squeeze())))
# print(compute_spectral_sdr(gt_xs[1].squeeze(), torch.rand_like(gt_xs[1].squeeze())))
#
# print(compute_spectral_sdr(gt_xs[0].squeeze(), gt_xs[0].squeeze()))
# print(compute_spectral_sdr(gt_xs[1].squeeze(), gt_xs[1].squeeze()))


# print(compute_spectral_metrics(gt_xs, [separated_1, separated_2], phases))
