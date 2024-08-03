import json
import random

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from evaluation_metric_functions import compute_spectral_sdr, compute_spectral_metrics
from functions_prior import PriorDataset
from separate_new import separate, get_vaes

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

dataloaders_train = [DataLoader(train_datasets[i], batch_size=256, shuffle=True, num_workers=12) for i in range(4)]
dataloaders_val = [DataLoader(val_datasets[i], batch_size=256, shuffle=True, num_workers=12) for i in range(4)]


total_basis_sdr_1 = 0
total_basis_sdr_2 = 0

total_sample_sdr_1 = 0
total_sample_sdr_2 = 0
stem_indices = [0, 3]

for _ in range(1):
    gt_data = [val_datasets[stem_index][random.randint(0, 100)] for stem_index in stem_indices]
    gt_xs = [data['spectrogram'] for data in gt_data]

    gt_m = torch.sum(torch.cat(gt_xs), dim=0)

    # TODO: Allow for batch processing
    separated = separate(gt_m, name='toy')

    separated_1 = separated[0].detach().cpu().reshape((64, 64))
    separated_2 = separated[1].detach().cpu().reshape((64, 64))

    total_basis_sdr_1 += compute_spectral_sdr(gt_xs[0].squeeze(), separated_1)
    total_basis_sdr_2 += compute_spectral_sdr(gt_xs[1].squeeze(), separated_2)

    vaes = get_vaes(name, stem_indices)
    samples = [vae.decode(torch.randn(1, hps['hidden']).to(device)).squeeze().detach().cpu() for vae in vaes]

    sample_sdrs = np.zeros((k, k))

    total_sample_sdr_1 += compute_spectral_sdr(gt_xs[0].squeeze(), samples[0])
    total_sample_sdr_2 += compute_spectral_sdr(gt_xs[1].squeeze(), samples[1])

print(total_basis_sdr_1/20)
print(total_basis_sdr_2/20)
print(total_sample_sdr_1/20)
print(total_sample_sdr_2/20)


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
