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
from functions import set_seed
from functions_prior import PriorDataset
from separate_new import separate, get_vaes

set_seed(42)

def evaluate_basis_recon_weights(name, num_samples=128):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    np.set_printoptions(precision=3, suppress=True)

    # name = 'toy'
    hps = json.load(open(f'hyperparameters/{name}_stem1.json'))
    image_h = hps['image_h']
    image_w = hps['image_w']

    k = 2

    test_datasets = [
        PriorDataset('test', image_h=image_h, image_w=image_w, name='toy_dataset', num_stems=4, debug=False, stem_type=i + 1)
        for i in range(4)]

    test_data_size = len(test_datasets[0])

    metrics = {'sdr': 0,
               'isr': 1,
               'sir': 2,
               'sar': 3}

    weights = np.linspace(20, -20, num=40)
    basis = np.zeros((len(weights), k, len(metrics.keys()), num_samples))

    for weight_index, constraint_term_weight in enumerate(weights):
        for i in range(num_samples):
            stem_indices = [random.randint(0, 3), random.randint(0, 3)]
            print(i)
            gt_data = [test_datasets[stem_index][random.randint(0, test_data_size-1-k)+j] for j, stem_index in enumerate(stem_indices)]
            gt_xs = [data['spectrogram'] for data in gt_data]

            gt_m = torch.sum(torch.cat(gt_xs), dim=0)

            separated = separate(gt_m.to(device), hps, name=name, finetuned=False, alpha=1, visualise=False, verbose=False, constraint_term_weight=constraint_term_weight)

            separated_1 = separated[0].detach().cpu().reshape((64, 64))
            separated_2 = separated[1].detach().cpu().reshape((64, 64))

            time1 = time.time()
            sdr, isr, sir, sar, perm = mir_eval.separation.bss_eval_images(np.array([x.squeeze().view(-1) for x in gt_xs]), np.array([separated_1.view(-1), separated_2.view(-1)]), compute_permutation=True)
            time2 = time.time()
            # print(f'One metric took {time2 - time1} seconds')

            basis[weight_index, :, :, i] = np.array([sdr, isr, sir, sar]).T

        print(f'Weight: {constraint_term_weight}')
        print(f'{round(np.mean(basis[weight_index, 0, metrics["sdr"], :]), 3)} +- {round(np.std(basis[weight_index, 0, metrics["sdr"], :]), 3)}')
        print(f'{round(np.mean(basis[weight_index, 1, metrics["sdr"], :]), 3)} +- {round(np.std(basis[weight_index, 1, metrics["sdr"], :]), 3)}')
        print()

    np.save('results/weights_evaluated.npy', weights)
    np.save(f'results/basis_contrastive_weight_experiment_results_{name}.npy', basis)
