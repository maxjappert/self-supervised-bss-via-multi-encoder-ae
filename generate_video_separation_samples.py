import json
import random

import numpy as np
import torch
from matplotlib import pyplot as plt

from functions_prior import MultiModalDataset
from separate_video import separate_video

device = 'cuda'

hps_stems = json.load(open(f'hyperparameters/vn_vn.json'))
hps_video = json.load(open(f'hyperparameters/video_model_raft_resnet.json'))

dataset = MultiModalDataset('val', normalise=hps_video['normalise'], fps=hps_video['fps'])

def rotate_image(image):
    return np.rot90(image, 2)


def generate_video_separation_samples(num_samples=10):
    for sample_idx in range(num_samples):
        print(sample_idx)
        while True:
            sample = dataset[random.randint(0, len(dataset) - 1)]
            if sample['label'] == 1:
                break

        gt_xs = sample['sources'].to(device)
        video = sample['video'].unsqueeze(dim=0).to(device)

        gt_m = torch.sum(gt_xs, dim=0).to(device)

        # separate using basis
        separated_basis = separate_video(gt_m, None,  hps_stems, hps_video, 'video_model_raft_resnet', sample['stem_names'],
                                         alpha=1, visualise=False, verbose=False, gradient_weight=15,
                                         constraint_term_weight=-15, device=device)
        separated_basis = [x_i.detach().cpu().view((64, 64)) for x_i in separated_basis]

        separated_basis_video = separate_video(gt_m, video,  hps_stems, hps_video, 'video_model_raft_resnet', sample['stem_names'],
                                         alpha=1, visualise=False, verbose=False, gradient_weight=15,
                                         constraint_term_weight=-15, device=device, video_weight=1024)
        separated_basis_video = [x_i.detach().cpu().view((64, 64)) for x_i in separated_basis_video]

        gt_xs = np.array([x.detach().cpu().squeeze() for x in gt_xs])
        gt_m = gt_m.detach().cpu().squeeze()

        # Titles for columns
        titles = ["Mixture", "Source 1", "Source 2"]

        # Plotting
        fig, axes = plt.subplots(3, 3, figsize=(9, 9))

        # Row titles
        row_titles = ["Ground Truth", "Without Video", "With Video"]

        # Populate the figure
        for i, (mixture, source1, source2) in enumerate([
            (gt_m, gt_xs[0], gt_xs[1]),
            (np.sum(separated_basis, axis=0), separated_basis[0], separated_basis[1]),
            (np.sum(separated_basis_video, axis=0), separated_basis_video[0], separated_basis_video[1])
        ]):
            axes[i, 0].imshow(rotate_image(mixture), cmap='gray')
            axes[i, 1].imshow(rotate_image(source1), cmap='gray')
            axes[i, 2].imshow(rotate_image(source2), cmap='gray')

            for j in range(3):
                axes[i, j].axis('off')
                if i == 0:  # Set the column titles on the first row
                    axes[i, j].set_title(titles[j])

            # Set the row title on the first column
            axes[i, 0].text(-0.2, 0.5, row_titles[i], va='center', ha='right', fontsize=12, transform=axes[i, 0].transAxes)

        plt.savefig(f'figures/rochester_separation_{sample_idx+1}.png', dpi=300, bbox_inches='tight')
