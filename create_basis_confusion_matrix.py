import json
import sys

import matplotlib.pyplot as plt
import mir_eval
import numpy as np
import torch
from torch.utils.data import DataLoader
import pandas as pd

from functions import TwoSourcesDataset, set_seed
from functions_prior import PriorDataset
from separate_new import get_vaes, separate

import seaborn as sn

set_seed(42)

def create_basis_confusion_matrix(model_name, num_samples_per_combo=128):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model_name = sys.argv[1]
    hps = json.load(open(f'hyperparameters/{model_name}_stem1.json'))
    dataset_name = 'toy_dataset' if model_name.__contains__('toy') else 'musdb_18_prior'
    image_h = 64
    image_w = 64

    test_datasets = [PriorDataset('test', image_h=image_h, image_w=image_w, name=dataset_name, num_stems=4, debug=False,
                                  stem_type=i + 1) for i in range(4)]

    # test_datasets = [TwoSourcesDataset(split='test', debug=False, name='toy_dataset', reduction_ratio=0.0001)]

    vaes = get_vaes(model_name, [0, 1, 2, 3])

    confusion_matrix = np.zeros((4, 4))

    for vae_idx1, vae1 in enumerate(vaes):
        for vae_idx2, vae2 in enumerate(vaes):
            dataloader1 = DataLoader(test_datasets[vae_idx1], batch_size=num_samples_per_combo, shuffle=True, num_workers=4)
            dataloader2 = DataLoader(test_datasets[vae_idx2], batch_size=num_samples_per_combo, shuffle=True, num_workers=4)
            batch1 = next(iter(dataloader1))
            batch2 = next(iter(dataloader2))
            batch1 = batch1['spectrogram'].to(device).float()
            batch2 = batch2['spectrogram'].to(device).float()
            assert batch1.shape[0] == batch2.shape[0] == num_samples_per_combo

            batch_combined = batch1 + batch2

            for i in range(num_samples_per_combo):
                print(i)
                source1, source2 = separate(batch_combined[i], hps, name=model_name, sigma_end=0.5 if model_name.__contains__('toy') else 1.0, finetuned=True, visualise=False, verbose=False, k=2, constraint_term_weight=-18, stem_indices=[vae_idx1, vae_idx2])
                separated_basis = np.array([source1.detach().cpu().view(-1), source2.detach().cpu().view(-1)])
                gt = np.array([batch1[i].cpu().view(-1), batch2[i].cpu().view(-1)])
                sdr, isr, sir, sar, _ = mir_eval.separation.bss_eval_images(separated_basis, gt)

                confusion_matrix[vae_idx1, vae_idx2] += sdr.mean()

            confusion_matrix[vae_idx1, vae_idx2] /= num_samples_per_combo

    print(confusion_matrix)

    stems = ['Sine', 'Sawtooth', 'Square', 'Triangle'] if dataset_name == 'toy_dataset' else ['Drums', 'Bass', 'Other', 'Vocals']

    df_cm = pd.DataFrame(confusion_matrix, index=stems, columns=stems)
    # plt.figure(figsize=(10,7))
    sn.set(font_scale=1.4) # for label size
    sn.heatmap(df_cm, annot=False, annot_kws={"size": 16}) # font size

    plt.savefig(f'figures/confusion_matrix_basis_{model_name}.png')
