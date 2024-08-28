import json
import random

import numpy as np
import torch
from matplotlib import pyplot as plt

from functions_prior import MultiModalDataset, PriorDataset
from separate_new import get_vaes, separate
from separate_video import separate_video

device = 'cuda'
name_vae = 'toy'

dataset_name = 'toy_dataset' if name_vae.__contains__('toy') else 'musdb_18_prior'

row_labels = ['Ground Truth', 'Separated']

def rotate_image(image):
    return np.rot90(image, 2)

for sample_idx in range(10):

    stem_indices = [random.randint(0, 3), random.randint(0, 3)]

    hps_vae = json.load(open(f'hyperparameters/{name_vae}_stem1.json'))
    image_h = hps_vae['image_h']
    image_w = hps_vae['image_w']
    vaes = get_vaes(name_vae, stem_indices, device)

    # Create the plot
    fig, axs = plt.subplots(2, 3, figsize=(9, 6))


    # Define column titles
    column_titles = ['Mix', 'Source 1', 'Source 2']

    test_datasets = [
        PriorDataset('test', image_h=image_h, image_w=image_w, name=dataset_name, num_stems=4, debug=False,
                     stem_type=i + 1)
        for i in range(4)]

    gt_data = [test_datasets[stem_index][random.randint(0, 100)] for stem_index in stem_indices]
    gt_xs = [data['spectrogram'] for data in gt_data]

    gt_m = torch.sum(torch.cat(gt_xs), dim=0).to(device)

    separated_basis = separate(gt_m,
                               hps_vae,
                               name=name_vae,
                               finetuned=False,
                               alpha=1,
                               visualise=False,
                               verbose=False,
                               constraint_term_weight=-15,
                               stem_indices=stem_indices,
                               device=device,
                               gradient_weight=1)

    separated_basis = [rotate_image(x_i.detach().cpu().view(64, 64).numpy()) for x_i in separated_basis]

    gt_xs = [rotate_image(x_i.detach().cpu().squeeze().numpy()) for x_i in gt_xs]

    # Visualize the data
    axs[1, 0].imshow(np.sum(separated_basis, axis=0), cmap='gray')
    axs[1, 1].imshow(separated_basis[0], cmap='gray')
    axs[1, 2].imshow(separated_basis[1], cmap='gray')

    axs[0, 0].imshow(rotate_image(gt_m.cpu().view(64, 64).numpy()), cmap='gray')
    axs[0, 1].imshow(gt_xs[0], cmap='gray')
    axs[0, 2].imshow(gt_xs[1], cmap='gray')

    # Set column titles
    for ax, col in zip(axs[0], column_titles):
        ax.set_title(col)

    # Add row labels
    for i, label in enumerate(row_labels):
        axs[i, 0].text(-0.5, 0.5, label, va='center', ha='right', fontsize=12, transform=axs[i, 0].transAxes)


    # Remove axis labels
    for i in range(3):
        axs[0, i].axis('off')
        axs[1, i].axis('off')



    # Save the figure
    # plt.tight_layout()
    plt.savefig(f'figures/basis_separation_{name_vae}_{sample_idx+1}.png', dpi=300, bbox_inches='tight')
    # plt.show()
