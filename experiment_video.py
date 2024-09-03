import sys
from datetime import datetime
import json
import random

import mir_eval
import numpy as np
import torch
from functions_prior import PriorDataset, MultiModalDataset
from separate_video import separate_video

np.set_printoptions(precision=3, suppress=True)

def print_output(results, k, metric_idx, name, index):
    for stem_idx in range(k):
        print(
            f'{name} {stem_idx + 1}: {round(np.mean(results[stem_idx, metric_idx, :index+1]), 3)} +- {round(np.std(results[stem_idx, metric_idx, :index+1]), 3)}')

    print()


def experiment_video(name_video_model, num_samples=100, video_weight=1024, device='cuda'):
    k = 2
    image_h = 64
    image_w = 64

    # name_vae = sys.argv[1]
    hps_stems = json.load(open(f'hyperparameters/vn_vn.json'))
    hps_video = json.load(open(f'hyperparameters/{name_video_model}.json'))

    dataset = MultiModalDataset('val', normalise=hps_video['normalise'], fps=hps_video['fps'])

    metrics = {'sdr': 0,
               'isr': 1,
               'sir': 2,
               'sar': 3}

    results_basis = np.zeros((k, len(metrics.keys()), num_samples))
    results_basis_video = np.zeros((k, len(metrics.keys()), num_samples))

    for i in range(num_samples):
        while True:
            sample = dataset[random.randint(0, len(dataset)-1)]
            if sample['label'] == 1:
                break

        now = datetime.now()
        timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")
        print(f'[{timestamp_str}]  Processing {i+1}/{num_samples}')
        # to avoid, when the same stem is selected, the same sample
        gt_xs = sample['sources'].to(device)
        video = sample['video'].unsqueeze(dim=0).to(device)

        gt_m = torch.sum(gt_xs, dim=0).to(device)

        # separate using basis
        separated_basis = separate_video(gt_m, None,  hps_stems, hps_video, name_video_model, sample['stem_names'],
                                         alpha=1, visualise=False, verbose=False, device=device)
        separated_basis = [x_i.detach().cpu().view(-1) for x_i in separated_basis]

        separated_basis_video = separate_video(gt_m, video,  hps_stems, hps_video, name_video_model, sample['stem_names'],
                                         alpha=1, visualise=False, verbose=False, device=device, video_weight=video_weight)
        separated_basis_video = [x_i.detach().cpu().view(-1) for x_i in separated_basis_video]

        gt_xs = np.array([x.squeeze().detach().cpu().view(-1) for x in gt_xs])

        # compute metrics
        sdr_basis, isr_basis, sir_basis, sar_basis, _ = mir_eval.separation.bss_eval_images(gt_xs, separated_basis, compute_permutation=True)
        sdr_basis_video, isr_basis_video, sir_basis_video, sar_basis_video, _ = mir_eval.separation.bss_eval_images(gt_xs, separated_basis_video, compute_permutation=True)

        # put metrics into relevant location for evaluation later
        results_basis[:, :, i] = np.array([sdr_basis, isr_basis, sir_basis, sar_basis]).T
        results_basis_video[:, :, i] = np.array([sdr_basis_video, isr_basis_video, sir_basis_video, sar_basis_video]).T

        print_output(results_basis, k, 0, 'BASIS', i)
        print_output(results_basis_video, k, 0, 'BASIS Video', i)

        np.save(f'results/results_basis_rochester_novideo_{video_weight}.npy', results_basis)
        np.save(f'results/results_basis_rochester_video_{video_weight}.npy', results_basis_video)

       # print(f'Weight: {constraint_term_weight}')
        # print(f'{round(np.mean(basis_sdr_1), 3)} +- {round(np.std(basis_sdr_1), 3)}')
        # print(f'{round(np.mean(basis_sdr_2), 3)} +- {round(np.std(basis_sdr_2), 3)}')
        # print()


# experiment_video('video_model_raft_resnet', num_samples=50, device='cuda', gradient_weight=15, video_weight=64)
# experiment_video('video_model_raft_resnet', num_samples=50, device='cuda', gradient_weight=15, video_weight=20000)
