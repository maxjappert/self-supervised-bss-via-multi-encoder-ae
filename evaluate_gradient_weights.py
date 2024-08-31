import json
import random
import time
from datetime import datetime

import mir_eval
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from evaluation_metric_functions import compute_spectral_sdr, compute_spectral_metrics
from functions import set_seed
from functions_prior import PriorDataset, MultiModalDataset
from separate_new import separate, get_vaes
from separate_video import separate_video

set_seed(42)

def evaluate_gradient_weight_video(name, num_samples=128, name_video_model='video_model_raft_resnet'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # name_vae = sys.argv[1]
    hps_stems = json.load(open(f'hyperparameters/vn_vn.json'))
    hps_video = json.load(open(f'hyperparameters/{name_video_model}.json'))

    dataset = MultiModalDataset('val', normalise=hps_video['normalise'], fps=hps_video['fps'])

    metrics = {'sdr': 0,
               'isr': 1,
               'sir': 2,
               'sar': 3}

    weights = [i for i in range(1, 20)] # np.linspace(20, -20, num=40)
    basis = np.zeros((len(weights), 2, len(metrics.keys()), num_samples))

    for weight_index, gradient_weight in enumerate(weights):
        for i in range(num_samples):
            while True:
                sample = dataset[random.randint(0, len(dataset) - 1)]
                if sample['label'] == 1:
                    break

            now = datetime.now()
            timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")
            # print(f'[{timestamp_str}]  Processing {i + 1}/{num_samples}')
            # to avoid, when the same stem is selected, the same sample
            gt_xs = sample['sources'].to(device)
            video = sample['video'].unsqueeze(dim=0).to(device)

            gt_m = torch.sum(gt_xs, dim=0).to(device)

            separated_basis_video = separate_video(gt_m, None, hps_stems, hps_video, name_video_model,
                                                   sample['stem_names'],
                                                   alpha=1, visualise=False, verbose=False,
                                                   constraint_term_weight=-15, gradient_weight=gradient_weight, device=device)
            separated_basis_video = [x_i.detach().cpu().view(-1) for x_i in separated_basis_video]

            sdr, isr, sir, sar, perm = mir_eval.separation.bss_eval_images(np.array([x.squeeze().detach().cpu().view(-1) for x in gt_xs]), separated_basis_video, compute_permutation=True)
            # print(f'One metric took {time2 - time1} seconds')

            basis[weight_index, :, :, i] = np.array([sdr, isr, sir, sar]).T

        print(f'Weight: {gradient_weight}')
        print(f'{round(np.mean(basis[weight_index, 0, metrics["sdr"], :]), 3)} +- {round(np.std(basis[weight_index, 0, metrics["sdr"], :]), 3)}')
        print(f'{round(np.mean(basis[weight_index, 1, metrics["sdr"], :]), 3)} +- {round(np.std(basis[weight_index, 1, metrics["sdr"], :]), 3)}')
        print()

        np.save('results/gradient_video_weights_evaluated.npy', weights)
        np.save(f'results/gradient_video_weights.npy', basis)

def evaluate_gradient_weight(name, num_samples=128):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    hps_vae = json.load(open(f'hyperparameters/{name}_stem1.json'))
    dataset_name = 'toy_dataset' if name.__contains__('toy') else 'musdb_18_prior'

    test_datasets = [
        PriorDataset('test', image_h=64, image_w=64, name=dataset_name, num_stems=4, debug=False, stem_type=i + 1)
        for i in range(4)]

    metrics = {'sdr': 0,
               'isr': 1,
               'sir': 2,
               'sar': 3}

    weights = [i for i in range(20)] # np.linspace(20, -20, num=40)
    separated_basis = np.zeros((len(weights), 2, len(metrics.keys()), num_samples))

    for weight_index, gradient_weight in enumerate(weights):
        for i in range(num_samples):

            stem_indices = [random.randint(0, 3), random.randint(0, 3)]

            # vaes = get_vaes(name, stem_indices, device)

            sample = [test_datasets[stem_idx]['spectrogram'] for stem_idx in stem_indices]

            now = datetime.now()
            timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")
            # print(f'[{timestamp_str}]  Processing {i + 1}/{num_samples}')
            # to avoid, when the same stem is selected, the same sample
            gt_xs = torch.tensor(sample).to(device)

            gt_m = torch.sum(gt_xs, dim=0).to(device)

            # separate using basis
            separated_basis = separate(gt_m, hps_vae, name=name, stem_indices=stem_indices, finetuned=False,
                                       alpha=1, visualise=False, verbose=False,
                                       constraint_term_weight=-15, gradient_weight=gradient_weight)
            separated_basis = [x_i.detach().cpu() for x_i in separated_basis]

            sdr, isr, sir, sar, perm = mir_eval.separation.bss_eval_images(np.array([x.squeeze().detach().cpu().view(-1) for x in gt_xs]), separated_basis, compute_permutation=True)
            # print(f'One metric took {time2 - time1} seconds')

            separated_basis[weight_index, :, :, i] = np.array([sdr, isr, sir, sar]).T

        print(f'Weight: {gradient_weight}')
        print(f'{round(np.mean(separated_basis[weight_index, 0, metrics["sdr"], :]), 3)} +- {round(np.std(separated_basis[weight_index, 0, metrics["sdr"], :]), 3)}')
        print(f'{round(np.mean(separated_basis[weight_index, 1, metrics["sdr"], :]), 3)} +- {round(np.std(separated_basis[weight_index, 1, metrics["sdr"], :]), 3)}')
        print()

        np.save(f'results/gradient_{name}_weights_evaluated.npy', weights)
        np.save(f'results/gradient_{name}_weights.npy', separated_basis)

