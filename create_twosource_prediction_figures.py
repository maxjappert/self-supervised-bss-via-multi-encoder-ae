import json

import torch
from matplotlib import pyplot as plt
from scipy.ndimage import rotate
from torch._dynamo.trace_rules import np

from evaluate_nmf import nmf_approx_two_sources
from functions import get_model, get_non_linear_separation, get_linear_separation
from functions_prior import PriorDataset
from separate_new import separate, get_vaes

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

k = 2


# stem_indices = [0, 2]

def create_pred_figures(name_vae, device):
    def create_pred_figure(stem_indices, name_vae, device):
        name_separator = f'{name_vae}_separator'
        hps_vae = json.load(open(f'hyperparameters/{name_vae}_stem1.json'))
        image_h = hps_vae['image_h']
        image_w = hps_vae['image_w']
        vaes = get_vaes(name_vae, stem_indices, device)

        dataset_name = 'toy_dataset' if name_vae.__contains__('toy') else 'musdb_18_prior'

        model_bss = get_model(name_separator, image_h=64, image_w=64, k=2)
        model_bss_linear = get_model(f'{name_separator}_linear', image_h=64, image_w=64, k=2)

        test_datasets = [
            PriorDataset('test', image_h=image_h, image_w=image_w, name=dataset_name, num_stems=4, debug=False, stem_type=i + 1)
            for i in range(4)]

        gt_data = [test_datasets[stem_index][17] for stem_index in stem_indices]
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
        separated_basis = [x_i.detach().cpu().view(64, 64).numpy() for x_i in separated_basis]

        hps_vae_finetuned = json.load(open(f'hyperparameters/toy_stem1.json'))

        separated_basis_finetuned = separate(gt_m,
                                             hps_vae_finetuned,
                                             sigma_end=0.5,
                                             name='toy',
                                             finetuned=True,
                                             alpha=1,
                                             visualise=False,
                                             verbose=False,
                                             constraint_term_weight=-18,
                                             stem_indices=stem_indices,
                                             device=device,
                                             gradient_weight=1)
        separated_basis_finetuned = [x_i.detach().cpu().view(64, 64).numpy() for x_i in separated_basis_finetuned]

        samples = [vae.decode(torch.randn(1, hps_vae['hidden']).to(device)).squeeze().detach().cpu() for vae in vaes]
        samples = [x_i.detach().cpu().view(64, 64).numpy() for x_i in samples]

        noise_images = [torch.rand_like(torch.tensor(samples[i])).numpy() for i in range(k)]

        nmf_recons = nmf_approx_two_sources(gt_m)
        nmf_recons = [torch.tensor(x_i).view(64, 64).numpy() for x_i in nmf_recons]

        # get vae_bss separation
        separated_bss = get_non_linear_separation(model_bss, gt_m.unsqueeze(0).unsqueeze(0))
        separated_bss = [torch.tensor(x_i).view(64, 64).numpy() for x_i in separated_bss]

        separated_bss_linear = get_linear_separation(model_bss_linear, gt_m.unsqueeze(0).unsqueeze(0))
        separated_bss_linear = [torch.tensor(x_i).view(64, 64).numpy() for x_i in separated_bss_linear]

        separated = np.stack([separated_basis, nmf_recons, separated_bss, separated_bss_linear, separated_basis_finetuned])

        fig, axs = plt.subplots(6, 3, figsize=(6, 10))

        gt = [gt_m] + gt_xs
        titles = ['Mixture', 'Source 1', 'Source 2']
        for i in range(3):
            axs[0, i].imshow(rotate(gt[i].squeeze().cpu(), angle=180), cmap='grey')
            axs[0, i].set_title(titles[i])
            axs[0, i].set_xticks([])
            axs[0, i].set_yticks([])

        for row_idx in range(5):
            images = [np.sum(separated[row_idx, :, :, :], axis=0), separated[row_idx, 0, :, :], separated[row_idx, 1, :, :]]

            for i in range(3):
                axs[row_idx+1, i].imshow(rotate(images[i], angle=180), cmap='grey')
                axs[row_idx + 1, i].set_xticks([])
                axs[row_idx + 1, i].set_yticks([])
                # axs[row_idx+1, i].set_title('title')
                # axs[row_idx+1, i].set_axis_off()

        axs[0, 0].set_ylabel('True')
        axs[1, 0].set_ylabel('BASIS')
        axs[2, 0].set_ylabel('NMF')
        axs[3, 0].set_ylabel('AE-BSS')
        axs[4, 0].set_ylabel('Linear AE-BSS')
        axs[5, 0].set_ylabel('BASIS Finetuned')
        # axs[6, 0].set_ylabel('BASIS Optimised')

        # plt.tight_layout()
        plt.savefig(f'figures/separation_2s_{name_vae}_{stem_indices[0]}_{stem_indices[1]}.png')

    for i in range(4):
        for j in range(4):
            if i > j:
                continue

            create_pred_figure([i, j], name_vae, device)
