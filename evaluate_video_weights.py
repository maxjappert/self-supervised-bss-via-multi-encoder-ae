import json
import random
import time
from datetime import datetime

import mir_eval
import numpy as np
import torch
from jinja2.filters import do_round
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from evaluation_metric_functions import compute_spectral_sdr, compute_spectral_metrics
from functions import set_seed
from functions_prior import PriorDataset, MultiModalDataset
from separate_video import separate_video

def evaluate_video_weight(num_samples=10, name_video_model='video_model_raft_resnet', device='cuda'):
    # name_vae = sys.argv[1]
    hps_stems = json.load(open(f'hyperparameters/vn_vn.json'))
    hps_video = json.load(open(f'hyperparameters/{name_video_model}.json'))

    dataset = MultiModalDataset('val', normalise=hps_video['normalise'], fps=hps_video['fps'])

    metrics = {'sdr': 0,
               'isr': 1,
               'sir': 2,
               'sar': 3}

    weights = [2**i for i in range(8)] # np.linspace(20, -20, num=40)
    basis = np.zeros((len(weights), 2, len(metrics.keys()), num_samples))

    for i in range(num_samples):
        for weight_index, video_weight in enumerate(weights):
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

            separated_basis_video = separate_video(gt_m, video, hps_stems, hps_video, name_video_model,
                                                   sample['stem_names'],
                                                   alpha=1, visualise=False, verbose=False, device=device, video_weight=video_weight)
            separated_basis_video = [x_i.detach().cpu().view(-1) for x_i in separated_basis_video]

            gt_xs = np.array([x.squeeze().detach().cpu().view(-1) for x in gt_xs])

            sdr, isr, sir, sar, perm = mir_eval.separation.bss_eval_images(gt_xs, separated_basis_video, compute_permutation=True)
            # print(f'One metric took {time2 - time1} seconds')

            basis[weight_index, :, :, i] = np.array([sdr, isr, sir, sar]).T

    np.save(f'results/video_weights_evaluated.npy', weights)
    np.save(f'results/video_weights.npy', basis)

    for weight_index, video_weight in enumerate(weights):
        print(f'Weight: {video_weight}')
        print(f'{round(np.mean(basis[weight_index, 0, metrics["sdr"], :]), 3)} +- {round(np.std(basis[weight_index, 0, metrics["sdr"], :], ddof=1), 3)}')
        print(f'{round(np.mean(basis[weight_index, 1, metrics["sdr"], :]), 3)} +- {round(np.std(basis[weight_index, 1, metrics["sdr"], :], ddof=1), 3)}')
        print()


# evaluate_video_weight(num_samples=50, device='cuda')
